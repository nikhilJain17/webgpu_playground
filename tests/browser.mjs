import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import { spawn } from 'node:child_process';
import assert from 'node:assert/strict';
const port = 8093;
const server = spawn('python3', ['-m','http.server',String(port),'--bind','127.0.0.1'], {stdio:'ignore'});
let browser;
try {
  for(let i=0;i<50;i++){try{const r=await fetch(`http://127.0.0.1:${port}/`);if(r.ok)break;}catch{} await new Promise(r=>setTimeout(r,100));}
  browser=await chromium.launch({executablePath:process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu','--use-angle=metal']});
  const page=await browser.newPage(); const pageErrors=[]; page.on('pageerror',e=>pageErrors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(()=>document.getElementById('status').textContent==='WebGPU ready',{},{timeout:20000});
  const report=await page.evaluate(async()=>{
    const {KernelRunner}=await import('/runner.js');
    const {reflect}=await import('/core.js');
    const base=await (await fetch('/configs/copy.json')).json();
    const results=[]; const runner=new KernelRunner(); await runner.init();
    const info=runner.device.adapterInfo;
    const gpu={vendor:info?.vendor,architecture:info?.architecture,device:info?.device,description:info?.description};
    const code=(w=256,debug=false)=>`@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
${debug?'@group(0) @binding(7) var<storage,read_write> debug:array<f32>;':''}
@compute @workgroup_size(${w}) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&input)){ output[id.x]=input[id.x];${debug?'debug[id.x]=input[id.x]*2.0;':''}}}`;
    const test=async(name,fn)=>{try{await fn();results.push({name,passed:true});}catch(e){results.push({name,passed:false,error:e.stack||e.message});}};
    const expect=(v,m)=>{if(!v)throw Error(m);};
    const fail=async(source,c,stage)=>{try{await runner.run(source,c);throw Error('Unexpected success');}catch(e){expect(e.stage===stage,`Expected ${stage}, got ${e.stage}: ${e.message}`);}};
    for(const w of [32,64,128,256]) for(const n of [1,16,255,256,257,1000]) await test(`copy N=${n}, WG=${w}, bindings 0/2`,async()=>{
      const c=structuredClone(base);c.parameters.N=n;const r=await runner.run(code(w),c);
      expect(r.check.passed,'Copy mismatch');expect(r.dispatch[0]===Math.ceil(n/w),'Dispatch mismatch');expect(r.outputs['0:2'].length===n,'Wrong length');
    });
    await test('syntax diagnostics and successful recovery',async()=>{await fail(code()+'\nnot valid WGSL',base,'shader diagnostics');expect((await runner.run(code(),base)).check.passed,'Recovery failed');});
    await test('missing binding rejected',async()=>{const c=structuredClone(base);c.resources.pop();await fail(code(),c,'configuration / bindings');});
    await test('extra binding 1 rejected',async()=>{const c=structuredClone(base);c.resources.push({...c.resources[0],binding:1});await fail(code(),c,'configuration / bindings');});
    await test('incorrect binding access rejected',async()=>{const c=structuredClone(base);c.resources[0].access='read_write';await fail(code(),c,'configuration / bindings');});
    await test('real pipeline validation failure',async()=>{await fail(code(65536),base,'pipeline validation');});
    await test('real binding validation error scope',async()=>{
      const d=runner.device;let caught=false;
      try{await runner.scope(d,'binding probe',()=>{const layout=d.createBindGroupLayout({entries:[{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}}]});d.createBindGroup({layout,entries:[]});});}catch(e){caught=e.stage==='binding probe';} expect(caught,'Validation error not captured');
    });
    await test('debug readback with checks disabled',async()=>{
      const c=structuredClone(base);c.check.mode='disabled';c.resources.push({...c.resources[1],binding:7,role:'debug'});
      const r=await runner.run(code(256,true),c);expect(r.check===null,'Check not disabled');expect(r.outputs['0:7'].every((v,i)=>v===r.outputs['0:2'][i]*2),'Wrong debug data');
    });
    await test('uniform packed fields, arbitrary binding/group, unused declarations',async()=>{
      const source=`struct Params { multiplier:f32, }; @group(1) @binding(5) var<uniform> p:Params;
@group(1) @binding(9) var<storage,read> a:array<f32>; @group(1) @binding(3) var<storage,read_write> b:array<f32>;
@group(0) @binding(10) var<storage,read> unused:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&a)){b[id.x]=a[id.x]*p.multiplier;}}`;
      const c=structuredClone(base);c.resources[0].group=1;c.resources[0].binding=9;c.resources[1].group=1;c.resources[1].binding=3;
      c.resources.push({group:1,binding:5,kind:'uniform',access:'read',role:'uniform',type:'u32',length:4,init:'fields',fields:[{offset:0,type:'f32',value:1}],readback:false});
      c.check.input='1:9';c.check.output='1:3';expect((await runner.run(source,c)).check.passed,'Uniform/sparse group copy mismatch');
    });
    await test('explicit XYZ dispatch and multidimensional elementwise rejection',async()=>{
      const source=`@group(0) @binding(8) var<storage,read_write> o:array<u32>; @compute @workgroup_size(2,2,1) fn main(@builtin(global_invocation_id) id:vec3u){o[id.x+4u*id.y+16u*id.z]=1u;}`;
      const c={entryPoint:'main',parameters:{N:32},resources:[{group:0,binding:8,kind:'storage',access:'read_write',role:'debug',type:'u32',length:32,init:'zero',readback:true}],dispatch:{mode:'explicit',workgroups:[2,2,2]},check:{mode:'disabled'}};
      expect((await runner.run(source,c)).outputs['0:8'].every(v=>v===1),'XYZ did not cover buffer');
      c.dispatch.mode='elementwise';await fail(source,c,'configuration / bindings');
    });
    await test('numerical failure distinct from execution success',async()=>{const r=await runner.run(code().replace('output[id.x]=input[id.x]','output[id.x]=input[id.x]+1.0'),base);expect(!r.check.passed && r.check.count===base.parameters.N,'Wrong mismatch result');});
    await test('nonfinite values reported',async()=>{const c=structuredClone(base);c.resources[0].init='zero';const r=await runner.run(code().replace('output[id.x]=input[id.x]','output[id.x]=input[id.x]/input[id.x]'),c);expect(r.check.nonfinite===c.parameters.N,'Nonfinite values not reported');});
    await test('overlap refused; repeated runs release all buffers',async()=>{
      const d=runner.device;const create=d.createBuffer.bind(d);let allocations=0,destroys=0;
      d.createBuffer=descriptor=>{allocations++;const b=create(descriptor);const destroy=b.destroy.bind(b);b.destroy=()=>{destroys++;destroy();};return b;};
      try{const pending=runner.run(code(),base);await fail(code(),base,'execution lock');await pending;for(let i=0;i<10;i++)await runner.run(code(),base);expect(allocations===destroys,`${allocations} allocations, ${destroys} destroys`);}finally{d.createBuffer=create;}
    });
    await test('submission scope catches invalid command buffer',async()=>{
      const d=runner.device;let caught=false;
      try{await runner.scope(d,'command submission',()=>{const e=d.createCommandEncoder();const p=e.beginComputePass();p.dispatchWorkgroups(1);p.end();d.queue.submit([e.finish()]);});}catch(e){caught=e.stage==='command submission';}expect(caught,'Submission validation missed');
    });
    await test('readback rejection clears run and releases buffers',async()=>{
      const d=runner.device;const create=d.createBuffer.bind(d);let allocations=0,destroys=0;
      d.createBuffer=desc=>{const b=create(desc);allocations++;const destroy=b.destroy.bind(b);b.destroy=()=>{destroys++;destroy();};if(desc.usage & GPUBufferUsage.MAP_READ)b.mapAsync=()=>Promise.reject(Error('Injected readback failure'));return b;};
      try{await fail(code(),base,'readback');expect(allocations===destroys,'Failure leaked buffers');}finally{d.createBuffer=create;}
      expect((await runner.run(code(),base)).check.passed,'Recovery failed');
    });
    await test('override workgroup uses explicit dispatch and configured constants',async()=>{
      const c=structuredClone(base);c.parameters.N=257;c.dispatch={mode:'explicit',workgroups:[2,1,1]};c.constants={W:256};
      const source='override W:u32=32u;\n'+code('W');
      expect((await runner.run(source,c)).check.passed,'Override copy failed');
      c.dispatch.mode='elementwise';await fail(source,c,'configuration / bindings');
    });
    await test('selected entry point includes helper resources only',async()=>{
      const source=code().replace('output[id.x]=input[id.x];','copyValue(id.x);')+'\nfn copyValue(i:u32){output[i]=input[i];} @group(0) @binding(6) var<storage,read_write> other:array<f32>; @compute @workgroup_size(1) fn second(){other[0]=1.0;}';
      expect((await runner.run(source,base)).check.passed,'Selected entry layout mismatch');
    });
    await test('uncaptured GPU validation fallback invalidates device session',async()=>{
      let fatal; const r=new KernelRunner(m=>fatal=m);await r.init();
      r.device.createBuffer({size:4,usage:0});await new Promise(resolve=>setTimeout(resolve,100));
      expect(fatal?.includes('Uncaptured GPU error'),'Fallback listener missed error');
      let rejected=false;try{await r.run(code(),base);}catch{rejected=true;}expect(rejected,'Fatal error silently cleared');r.device.destroy();
    });
    await test('device loss reported and a new device can run',async()=>{
      let loss;const r=new KernelRunner(m=>loss=m);await r.init();r.device.destroy();await new Promise(resolve=>setTimeout(resolve,100));expect(loss?.includes('Device loss'),'No loss notification');expect((await r.run(code(),base)).check.passed,'Reinitialization failed');r.device.destroy();
    });
    runner.device.destroy();return {gpu,results};
  });
  // Exercise the real UI, including stale-output clearing and source identity.
  await page.locator('#editor').fill(`@group(0) @binding(0) var<storage,read> a:array<f32>; @group(0) @binding(2) var<storage,read_write> b:array<f32>; @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&a)){b[id.x]=a[id.x];}}`);
  await page.locator('#check').selectOption('copy');await page.locator('#run').click();await page.waitForFunction(()=>document.getElementById('execution').textContent.includes('SUCCESS'));
  assert.match(await page.locator('#correctness').textContent(),/PASS/);assert.match(await page.locator('#source-status').textContent(),/UNSAVED/);
  await page.locator('#editor').fill('invalid shader');await page.locator('#run').click();await page.waitForFunction(()=>document.getElementById('execution').textContent.includes('FAILED'));
  assert.equal(await page.locator('#outputs').textContent(),'');assert.match(await page.locator('#correctness').textContent(),/NOT CHECKED/);
  report.results.push({name:'UI source identity, copy pass, syntax failure clears stale results',passed:true});
  report.pageErrors=pageErrors;report.browser=await browser.version();
  await fs.mkdir('test-results',{recursive:true});await fs.writeFile('test-results/browser.json',JSON.stringify(report,null,2));await page.screenshot({path:'test-results/browser.png',fullPage:true});
  console.log(JSON.stringify({browser:report.browser,gpu:report.gpu,passed:report.results.filter(r=>r.passed).length,total:report.results.length,failures:report.results.filter(r=>!r.passed),pageErrors},null,2));
  assert.equal(report.results.filter(r=>!r.passed).length,0,'Browser tests failed');assert.deepEqual(pageErrors,[]);
}finally{await browser?.close();server.kill();}
