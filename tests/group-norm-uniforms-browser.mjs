// Execute an opaque, unchanged external source snapshot with three configurations.
// Kernel validation/numerical failures are saved, not repaired or treated as harness passes.
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const sourcePath='../stable-diffusion.cpp/ggml/src/ggml-webgpu/wgsl-shaders/group_norm.wgsl';
const source=await fs.readFile(sourcePath,'utf8');
const sha256=createHash('sha256').update(source).digest('hex');
const cases=await Promise.all([3,8,10].map(async size=>({size,config:JSON.parse(await fs.readFile(`configs/group-norm-uniform-${size}.json`,'utf8'))})));
const server=spawn('python3',['-m','http.server','8095','--bind','127.0.0.1'],{stdio:'ignore'});
let browser;
try {
  for(let i=0;i<50;i++){try{if((await fetch('http://127.0.0.1:8095/')).ok)break;}catch{} await new Promise(r=>setTimeout(r,100));}
  browser=await chromium.launch({executablePath:process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu','--use-angle=metal']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const pageErrors=[];page.on('pageerror',e=>pageErrors.push(e.message));
  await page.goto('http://127.0.0.1:8095/?config=group-norm-uniform-3&source=external/stable-diffusion-group-norm.wgsl');
  await page.waitForFunction(()=>document.getElementById('status').textContent==='WebGPU ready',null,{timeout:20000});
  assert.equal(await page.locator('#editor').inputValue(),source);
  await fs.mkdir('test-results',{recursive:true});
  const ui=[];
  for(const {size} of cases){
    await page.locator('#import').setInputFiles(`configs/group-norm-uniform-${size}.json`);
    await page.waitForFunction(size=>JSON.parse(document.getElementById('parameters').value).spatial[0]===size,size);
    assert.equal(await page.locator('#constants').inputValue(),'{"WORKGROUP_SIZE":4}');
    assert.equal(await page.locator('#dispatch').inputValue(),'explicit');
    await page.locator('#run').click();
    await page.waitForFunction(()=>/SUCCESS|FAILED/.test(document.getElementById('execution').textContent));
    const snapshot={size};
    for(const id of ['executed','execution','correctness','errors','outputs']) snapshot[id]=await page.locator(`#${id}`).textContent();
    assert.match(snapshot.executed,new RegExp(sha256));
    if(snapshot.execution.includes('SUCCESS')){
      assert.match(snapshot.execution,/2 × 1 × 1/);
      assert.match(snapshot.outputs,new RegExp(`Binding 0:2 — ${size*2} values`));
    }else{
      assert.equal(snapshot.outputs,'');assert.match(snapshot.correctness,/NOT CHECKED/);
    }
    ui.push(snapshot);
    await page.screenshot({path:`test-results/group-norm-uniform-${size}.png`,fullPage:true});
    assert.equal(await page.locator('#editor').inputValue(),source);
  }
  const probe=await page.evaluate(async({source,cases})=>{
    const {KernelRunner}=await import('/runner.js');
    const {parameters,bufferData,groupNorm}=await import('/core.js');
    const runner=new KernelRunner();await runner.init();
    const info=runner.device.adapterInfo;
    const results=[];
    try{
      for(const {size,config} of cases){
        const p=parameters(config.parameters),input=bufferData(config.resources[0],p);
        const reference=groupNorm(input,config.parameters),uniform=bufferData(config.resources[1],p);
        const record={size,N:p.N,input:Array.from(input),uniformBytes:Array.from(new Uint8Array(uniform.buffer)),means:reference.means,variances:reference.variances,expected:Array.from(reference.output),runs:[]};
        for(let i=0;i<5;i++){
          try{
            const r=await runner.run(source,config);
            record.runs.push({execution:'SUCCESS',correctness:r.check.passed?'PASS':'FAIL',actual:Array.from(r.outputs['0:2']),uniformReadback:Array.from(r.outputs['0:1']),check:r.check,diagnostics:r.diagnostics});
          }catch(e){record.runs.push({execution:'FAILED',stage:e.stage,error:e.message,diagnostics:e.diagnostics});}
        }
        results.push(record);
      }
    }finally{runner.device.destroy();}
    return {gpu:{vendor:info.vendor,architecture:info.architecture},results};
  },{source,cases});
  const report={sourcePath,sha256,browser:await browser.version(),cases,...probe,ui,pageErrors};
  await fs.writeFile('test-results/group-norm-uniforms.json',JSON.stringify(report,(_,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v,2));
  for(const r of probe.results){
    assert.deepEqual(r.means,[(r.size+1)/2,(3*r.size+1)/2]);
    assert.ok(r.variances.every(v=>Math.abs(v-(r.size*r.size-1)/12)<1e-12));
    const bytes=new Uint8Array(r.uniformBytes);const view=new DataView(bytes.buffer);
    assert.equal(bytes.length,16);assert.equal(view.getUint32(0,true),r.size);assert.equal(view.getFloat32(4,true),Math.fround(1e-6));
    const config=cases.find(c=>c.size===r.size).config;
    for(const run of r.runs){
      if(run.execution==='FAILED')continue;
      assert.equal(run.actual.length,r.N);
      assert.deepEqual(run.uniformReadback,Array.from(new Uint32Array(bytes.buffer)));
      const mismatches=run.actual.flatMap((a,i)=>!Number.isFinite(a)||Math.abs(a-r.expected[i])>config.check.atol+config.check.rtol*Math.abs(r.expected[i])?[i]:[]);
      assert.deepEqual(run.check.mismatches.map(m=>m.index),mismatches);
      assert.equal(run.check.count,mismatches.length);
      assert.equal(run.check.nonfinite,run.actual.filter(a=>!Number.isFinite(a)).length);
    }
  }
  assert.deepEqual(pageErrors,[]);
  assert.equal(await fs.readFile(sourcePath,'utf8'),source,'External source changed during verification');
  console.log(JSON.stringify({sourcePath,sha256,browser:report.browser,gpu:report.gpu,results:probe.results.map(r=>({size:r.size,N:r.N,means:r.means,variances:r.variances,passed:r.runs.filter(x=>x.correctness==='PASS').length,numericalFailures:r.runs.filter(x=>x.correctness==='FAIL').length,executionFailures:r.runs.filter(x=>x.execution==='FAILED'),firstMismatch:r.runs.find(x=>x.correctness==='FAIL')?.check,ui:ui.find(x=>x.size===r.size).correctness}))},null,2));
}finally{await browser?.close();server.kill();}
