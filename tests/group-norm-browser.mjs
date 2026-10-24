// Integration probe of the user's external shader. Never rewrites shader source.
// Numerical failures are recorded as failures, but are not harness test failures:
// this exercise intentionally may contain an unfinished/racing implementation.
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const count = Number(process.argv[2] || 16);
assert.ok([16,20].includes(count), 'Choose the 16- or 20-element preset');
const preset = `group-norm-${count}`;
const width = count / 2;
const sourcePath = '../stable-diffusion.cpp/ggml/src/ggml-webgpu/wgsl-shaders/group_norm.wgsl';
const source = await fs.readFile(sourcePath, 'utf8');
const sha256 = createHash('sha256').update(source).digest('hex');
const config = JSON.parse(await fs.readFile(`configs/${preset}.json`, 'utf8'));
const server = spawn('python3', ['-m','http.server','8094','--bind','127.0.0.1'], {stdio:'ignore'});
let browser;
try {
  for(let i=0;i<50;i++){try{if((await fetch('http://127.0.0.1:8094/')).ok)break;}catch{} await new Promise(r=>setTimeout(r,100));}
  const served = await (await fetch('http://127.0.0.1:8094/external/stable-diffusion-group-norm.wgsl')).text();
  assert.equal(served,source,'Served shader differs from requested external source');
  browser = await chromium.launch({executablePath:process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu','--use-angle=metal']});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const pageErrors=[];page.on('pageerror',e=>pageErrors.push(e.message));
  await page.goto(`http://127.0.0.1:8094/?config=${preset}&source=external/stable-diffusion-group-norm.wgsl`);
  await page.waitForFunction(()=>document.getElementById('status').textContent==='WebGPU ready',null,{timeout:20000});
  assert.equal(await page.locator('#editor').inputValue(),source);
  assert.match(await page.locator('#source-status').textContent(),/stable-diffusion.cpp/);
  assert.equal(await page.locator('#dispatch').inputValue(),'explicit');
  assert.equal(await page.locator('#dx').inputValue(),'2');
  assert.equal(await page.locator('#check').inputValue(),'group-norm');
  await page.locator('#run').click();
  await page.waitForFunction(()=>/SUCCESS|FAILED/.test(document.getElementById('execution').textContent));
  const ui = {};
  for(const id of ['executed','execution','correctness','errors','outputs']) ui[id]=await page.locator(`#${id}`).textContent();
  assert.match(ui.executed,new RegExp(sha256));
  if (ui.execution.includes('SUCCESS')) {
    assert.match(ui.execution,/2 × 1 × 1/);
    assert.match(ui.execution,/4 × 1 × 1/);
    assert.match(ui.outputs,new RegExp(`Binding 0:2 — ${count} values`));
  } else {
    assert.equal(ui.outputs, '');
    assert.match(ui.correctness, /NOT CHECKED/);
  }
  await fs.mkdir('test-results',{recursive:true});
  await page.screenshot({path:`test-results/${preset}.png`,fullPage:true});
  const probe = await page.evaluate(async({source,config})=>{
    const {KernelRunner}=await import('/runner.js');
    const {groupNorm}=await import('/core.js');
    const runner=new KernelRunner();await runner.init();
    const info=runner.device.adapterInfo;
    const gpu={vendor:info.vendor,architecture:info.architecture};
    const reference=groupNorm(Float32Array.from(config.resources[0].values),config.parameters);
    const runs=[];
    try {
      for(let i=0;i<20;i++) {
        try {
          const result=await runner.run(source,config);
          runs.push({execution:'SUCCESS',correctness:result.check.passed?'PASS':'FAIL',actual:Array.from(result.outputs['0:2']),check:result.check,diagnostics:result.diagnostics});
        } catch(e) { runs.push({execution:'FAILED',stage:e.stage,error:e.message}); }
      }
    } finally {runner.device.destroy();}
    return {gpu,expected:Array.from(reference.output),means:reference.means,variances:reference.variances,runs};
  },{source,config});
  assert.equal(probe.expected.length,count);
  const means=[(width+1)/2,(3*width+1)/2], variance=(width*width-1)/12;
  assert.deepEqual(probe.means,means);assert.deepEqual(probe.variances,[variance,variance]);
  probe.expected.forEach((v,i)=>assert.ok(Math.abs(v-((i%width+1-means[0])/Math.sqrt(variance+config.parameters.epsilon)))<1e-12));
  for(const run of probe.runs) {
    if (run.execution === 'FAILED') continue;
    assert.equal(run.actual.length,count);
    const mismatches=run.actual.flatMap((actual,index)=>{
      const expected=probe.expected[index];return !Number.isFinite(actual)||Math.abs(actual-expected)>config.check.atol+config.check.rtol*Math.abs(expected)?[{index,actual,expected}]:[];
    });
    assert.equal(run.check.count,mismatches.length);
    assert.deepEqual(run.check.mismatches.map(m=>m.index),mismatches.map(m=>m.index));
    assert.equal(run.correctness,mismatches.length?'FAIL':'PASS');
  }
  if (ui.execution.includes('SUCCESS')) {
  // Verify this exact configuration rejects changed bindings and clears the visible previous result.
  await page.locator('.resource').nth(1).locator('[data-field="binding"]').fill('1');
  await page.locator('#run').click();
  await page.waitForFunction(()=>document.getElementById('execution').textContent.includes('FAILED'));
  assert.equal(await page.locator('#outputs').textContent(),'');
  assert.match(await page.locator('#errors').textContent(),/Binding 0:1/);
  assert.match(await page.locator('#correctness').textContent(),/NOT CHECKED/);
  }
  assert.equal(await fs.readFile(sourcePath,'utf8'),source,'External shader changed during test');
  assert.deepEqual(pageErrors,[]);
  const report={sourcePath,sha256,config,browser:await browser.version(),...probe,ui,pageErrors};
  await fs.writeFile(`test-results/${preset}.json`,JSON.stringify(report,(_,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v,2));
  console.log(JSON.stringify({sourcePath,sha256,browser:report.browser,gpu:report.gpu,uiExecution:ui.execution,uiCorrectness:ui.correctness,expected:probe.expected,passingRuns:probe.runs.filter(r=>r.correctness==='PASS').length,failingRuns:probe.runs.filter(r=>r.correctness==='FAIL').length,firstFailure:probe.runs.find(r=>r.correctness==='FAIL'),executionFailures:probe.runs.filter(r=>r.execution==='FAILED'),means:probe.means,variances:probe.variances},null,2));
} finally {await browser?.close();server.kill();}
