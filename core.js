import { WgslReflect, ResourceType } from './node_modules/wgsl_reflect/wgsl_reflect.module.js';

export const keyOf = r => `${r.group}:${r.binding}`;
export function integer(value, name, min = 1) {
  if (!Number.isSafeInteger(value) || value < min) throw Error(`${name} must be an integer ≥ ${min}`);
  return value;
}
export function parameters(p) {
  if ('batch' in p) {
    for (const k of ['batch', 'channels', 'groups']) integer(p[k], k);
    if (!Array.isArray(p.spatial) || !p.spatial.length) throw Error('spatial must contain positive dimensions');
    const spatialSize = p.spatial.reduce((a, b) => a * integer(b, 'spatial dimension'), 1);
    if (p.channels % p.groups) throw Error('channels must be divisible by groups');
    if (!Number.isFinite(p.epsilon) || p.epsilon <= 0) throw Error('epsilon must be positive and finite');
    return { ...p, spatialSize, N: integer(p.batch * p.channels * spatialSize, 'N') };
  }
  return { ...p, N: integer(p.N, 'N') };
}
export function reflect(code, entryPoint) {
  let reflection;
  try { reflection = new WgslReflect(code); }
  catch (e) { throw Error(`WGSL reflection could not parse this source: ${e.message || String(e)}. A browser-valid construct may be unsupported by wgsl_reflect; source was not changed.`); }
  const entry = reflection.entry.compute.find(e => e.name === entryPoint);
  if (!entry) throw Error(`No compute entry point "${entryPoint}"`);
  const resources = entry.resources.map(r => {
    if (![ResourceType.Storage, ResourceType.Uniform].includes(r.resourceType))
      throw Error(`${r.name}: only storage and uniform buffers are supported (no textures/samplers)`);
    return { group: r.group, binding: r.binding, name: r.name,
      kind: r.resourceType === ResourceType.Uniform ? 'uniform' : 'storage',
      access: r.resourceType === ResourceType.Uniform ? 'read' : r.access || 'read',
      size: r.size, type: r.type.getTypeName(), info: r };
  });
  // Overrides may affect workgroup dimensions. Refuse rather than silently use a default.
  const unresolved = entry.overrides.length > 0;
  return { resources, workgroup: unresolved ? null : entry.workgroupSize, entries: reflection.entry.compute.map(e => e.name) };
}
export function dispatchFor(config, reflected, limits = {}) {
  let groups;
  if (config.dispatch.mode === 'elementwise') {
    const w = reflected.workgroup;
    if (!w || !w.every(v => Number.isSafeInteger(v) && v > 0))
      throw Error('Elementwise needs a statically resolved workgroup size; override-dependent entry points require explicit dispatch');
    if (w[1] !== 1 || w[2] !== 1) throw Error('Elementwise supports only 1D workgroups (Y=Z=1). Use explicit workgroup count for 2D/3D.');
    groups = [Math.ceil(parameters(config.parameters).N / w[0]), 1, 1];
  } else if (config.dispatch.mode === 'explicit') groups = config.dispatch.workgroups;
  else throw Error('Choose elementwise or explicit dispatch');
  if (!Array.isArray(groups) || groups.length !== 3) throw Error('Dispatch needs X/Y/Z');
  groups.forEach((g, i) => {
    integer(g, `Dispatch ${'XYZ'[i]}`);
    if (g > (limits.maxComputeWorkgroupsPerDimension ?? 65535)) throw Error('Dispatch exceeds device workgroup limit');
  });
  return groups;
}
const arrays = { f32: Float32Array, u32: Uint32Array, i32: Int32Array };
export function bufferData(r, p) {
  const Type = arrays[r.type];
  if (!Type) throw Error(`${keyOf(r)}: type must be f32, u32 or i32`);
  const count = integer(typeof r.length === 'string' ? p[r.length] : r.length, `${keyOf(r)} length`);
  // Keep accidental allocations bounded; device limits are checked separately.
  if (count > 16 * 1024 * 1024) throw Error('Buffer exceeds playground limit of 16M scalar elements');
  const data = new Type(count);
  if (r.init === 'deterministic') for (let i = 0; i < count; i++) data[i] = ((i * 17 % 101) - 50) / 8;
  else if (r.init === 'values') {
    if (!Array.isArray(r.values) || r.values.length !== count || !r.values.every(Number.isFinite)) throw Error(`${keyOf(r)}: provide exactly ${count} finite values`);
    data.set(r.values);
  } else if (r.init !== 'zero' && r.init !== 'fields') throw Error(`${keyOf(r)}: unknown initializer`);
  if (r.init === 'fields') {
    if (!Array.isArray(r.fields)) throw Error('fields initializer requires fields [{offset,type,value}]');
    const view = new DataView(data.buffer);
    for (const f of r.fields) {
      const value = typeof f.value === 'string' ? p[f.value] : f.value;
      integer(f.offset, 'field byte offset', 0);
      if (f.offset % 4 || f.offset + 4 > data.byteLength || !Number.isFinite(value)) throw Error('Invalid field offset/value');
      if (f.type === 'u32') { integer(value, 'u32 field', 0); if (value > 0xffffffff) throw Error('u32 overflow'); view.setUint32(f.offset, value, true); }
      else if (f.type === 'i32') { if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647) throw Error('i32 overflow'); view.setInt32(f.offset, value, true); }
      else if (f.type === 'f32') view.setFloat32(f.offset, value, true);
      else throw Error('Field type must be f32, u32 or i32');
    }
  }
  return data;
}
export function prepare(config, reflected, limits = {}) {
  const p = parameters(config.parameters);
  const required = new Map(reflected.resources.map(r => [keyOf(r), r]));
  const seen = new Set();
  const buffers = config.resources.map(r => {
    integer(r.group, 'group', 0); integer(r.binding, 'binding', 0);
    const key = keyOf(r), expected = required.get(key);
    if (seen.has(key)) throw Error(`Duplicate binding ${key}`);
    seen.add(key);
    if (!expected) throw Error(`Binding ${key} is not used by entry point ${config.entryPoint}; remove it from configuration`);
    if (r.kind !== expected.kind || r.access !== expected.access) throw Error(`Binding ${key} requires ${expected.kind}, ${expected.access}; configured ${r.kind}, ${r.access}`);
    if (!['input', 'output', 'debug', 'uniform'].includes(r.role)) throw Error(`${key}: choose an explicit resource role`);
    const data = bufferData(r, p);
    const max = r.kind === 'uniform' ? limits.maxUniformBufferBindingSize : limits.maxStorageBufferBindingSize;
    if (data.byteLength < expected.size) throw Error(`${key}: ${data.byteLength} bytes is smaller than reflected minimum ${expected.size}`);
    if (data.byteLength > (max ?? Infinity) || data.byteLength > (limits.maxBufferSize ?? Infinity)) throw Error(`${key}: buffer exceeds device limit`);
    return { ...r, key, data };
  });
  for (const key of required.keys()) if (!seen.has(key)) throw Error(`Missing binding ${key}; configure its role and data`);
  const check = config.check;
  if (!['disabled', 'copy', 'group-norm'].includes(check.mode)) throw Error('Unknown expected-result check');
  if (check.mode !== 'disabled') {
    const input = buffers.find(b => b.key === check.input), output = buffers.find(b => b.key === check.output);
    if (!input || !output || input === output || input.role !== 'input' || output.role !== 'output' || !output.readback || input.type !== 'f32' || output.type !== 'f32' || input.data.length !== p.N || output.data.length !== p.N)
      throw Error('Checks need explicitly selected, distinct F32 input/output buffers of length N, with output readback enabled');
    if (check.mode === 'group-norm') {
      if (!('batch' in p)) throw Error('Group norm requires batch/channels/spatial/groups/epsilon');
      for (const t of ['atol', 'rtol']) if (!Number.isFinite(check[t]) || check[t] < 0) throw Error(`${t} must be finite and nonnegative`);
    }
  }
  return { p, buffers, dispatch: dispatchFor(config, reflected, limits) };
}
// Independent CPU reference: NCHW (flattened spatial), population variance, no affine transform.
export function groupNorm(input, raw) {
  const p = parameters(raw);
  if (input.length !== p.N) throw Error('Reference input length does not match dimensions');
  const width = p.channels / p.groups * p.spatialSize;
  const output = new Float64Array(p.N), means = [], variances = [];
  for (let start = 0; start < p.N; start += width) {
    let mean = 0;
    for (let i = 0; i < width; i++) mean += input[start + i];
    mean /= width;
    let variance = 0;
    for (let i = 0; i < width; i++) variance += (input[start + i] - mean) ** 2;
    variance /= width;
    means.push(mean); variances.push(variance);
    for (let i = 0; i < width; i++) output[start + i] = (input[start + i] - mean) / Math.sqrt(variance + p.epsilon);
  }
  return { output, means, variances };
}
export function compare(actual, expected, atol = 0, rtol = 0) {
  if (actual.length !== expected.length) throw Error('Comparison length mismatch');
  const mismatches = []; let count = 0, nonfinite = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i], e = expected[i], finite = Number.isFinite(a) && Number.isFinite(e);
    if (!finite) nonfinite++;
    if (!finite || Math.abs(a - e) > atol + rtol * Math.abs(e)) {
      count++; if (mismatches.length < 32) mismatches.push({ index: i, actual: a, expected: e, absoluteError: Math.abs(a - e) });
    }
  }
  return { passed: count === 0, count, nonfinite, mismatches };
}
