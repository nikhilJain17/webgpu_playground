import { reflect, prepare, groupNorm, compare } from './core.js';

export class KernelRunner {
  device = null;
  busy = false;
  fault = null;
  constructor(onFatal = () => {}) { this.onFatal = onFatal; }
  async init() {
    if (this.device) return;
    if (!navigator.gpu) throw Error('WebGPU unavailable. Use a WebGPU-enabled browser on localhost or HTTPS.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error('No WebGPU adapter available');
    const device = await adapter.requestDevice();
    this.device = device; this.fault = null;
    device.addEventListener('uncapturederror', event => {
      event.preventDefault();
      this.fault = `Uncaptured GPU error: ${event.error.message}`;
      this.onFatal(this.fault);
    });
    device.lost.then(info => {
      if (this.device !== device) return;
      this.device = null;
      this.fault = `Device loss (${info.reason}): ${info.message || 'device no longer available'}`;
      this.onFatal(this.fault);
    });
  }
  async scope(device, stage, fn) {
    for (const type of ['internal', 'out-of-memory', 'validation']) device.pushErrorScope(type);
    let result, failure;
    try { result = await fn(); } catch (e) { failure = e; }
    const errors = [];
    for (let i = 0; i < 3; i++) {
      try { const e = await device.popErrorScope(); if (e) errors.push(e.message); }
      catch (e) { errors.push(e.message); }
    }
    if (failure || errors.length || this.fault) {
      const e = Error([failure?.message, ...errors, this.fault].filter(Boolean).join('\n'));
      e.stage = stage; throw e;
    }
    return result;
  }
  async run(code, config, onStage = () => {}) {
    if (this.busy) throw Object.assign(Error('A run is already in progress'), { stage: 'execution lock' });
    this.busy = true;
    const owned = []; let stage = 'device initialization', diagnostics = [];
    const at = name => { stage = name; onStage(name); };
    try {
      at(stage); await this.init();
      const device = this.device;
      if (this.fault) throw Error(this.fault);
      at('shader diagnostics');
      const module = await this.scope(device, stage, async () => {
        const shader = device.createShaderModule({ code, label: 'Editor source snapshot' });
        const info = await shader.getCompilationInfo();
        diagnostics = [...info.messages].map(m => ({ type: m.type, line: m.lineNum, column: m.linePos, message: m.message }));
        if (diagnostics.some(m => m.type === 'error')) throw Error(diagnostics.map(m => `${m.type} ${m.line}:${m.column}: ${m.message}`).join('\n'));
        return shader;
      });
      at('reflection');
      const reflected = reflect(code, config.entryPoint);
      at('configuration / bindings');
      const plan = prepare(config, reflected, device.limits);
      at('pipeline validation');
      const pipeline = await this.scope(device, stage, () => device.createComputePipelineAsync({
        label: config.entryPoint, layout: 'auto', compute: { module, entryPoint: config.entryPoint, constants: config.constants || {} }
      }));
      at('buffer / binding validation');
      const { groups, resources } = await this.scope(device, stage, () => {
        const resources = plan.buffers.map(r => {
          const buffer = device.createBuffer({ label: r.key, size: r.data.byteLength,
            usage: (r.kind === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
          owned.push(buffer);
          device.queue.writeBuffer(buffer, 0, r.data);
          let readBuffer;
          if (r.readback) {
            readBuffer = device.createBuffer({ label: `${r.key} readback`, size: r.data.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            owned.push(readBuffer);
          }
          return { ...r, buffer, readBuffer };
        });
        const groups = [];
        const maxGroup = Math.max(-1, ...resources.map(r => r.group));
        for (let group = 0; group <= maxGroup; group++) {
          groups.push(device.createBindGroup({ label: `Group ${group}`, layout: pipeline.getBindGroupLayout(group),
            entries: resources.filter(r => r.group === group).map(r => ({ binding: r.binding, resource: { buffer: r.buffer } })) }));
        }
        return { groups, resources };
      });
      at('command submission');
      await this.scope(device, stage, async () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        groups.forEach((group, i) => pass.setBindGroup(i, group));
        pass.dispatchWorkgroups(...plan.dispatch); pass.end();
        for (const r of resources) if (r.readBuffer) encoder.copyBufferToBuffer(r.buffer, 0, r.readBuffer, 0, r.data.byteLength);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
      });
      at('readback');
      const outputs = await this.scope(device, stage, async () => {
        const results = {};
        for (const r of resources) if (r.readBuffer) {
          await r.readBuffer.mapAsync(GPUMapMode.READ);
          try { results[r.key] = new r.data.constructor(r.readBuffer.getMappedRange().slice(0)); }
          finally { r.readBuffer.unmap(); }
        }
        return results;
      });
      at('expected-result check');
      let check = null, reference = null;
      if (config.check.mode !== 'disabled') {
        const input = plan.buffers.find(r => r.key === config.check.input).data;
        reference = config.check.mode === 'copy' ? { output: input } : groupNorm(input, plan.p);
        check = compare(outputs[config.check.output], reference.output,
          config.check.mode === 'copy' ? 0 : config.check.atol, config.check.mode === 'copy' ? 0 : config.check.rtol);
      }
      if (this.fault) throw Error(this.fault);
      return { outputs, check, reference, diagnostics, dispatch: plan.dispatch, workgroup: reflected.workgroup };
    } catch (e) {
      e.stage ||= stage; e.diagnostics = diagnostics; throw e;
    } finally {
      for (const buffer of owned) buffer.destroy();
      this.busy = false;
    }
  }
}
