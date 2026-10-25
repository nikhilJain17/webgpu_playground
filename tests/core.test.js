import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parameters, groupNorm, compare, reflect, prepare, dispatchFor, bufferData } from '../core.js';
const copy = () => JSON.parse(fs.readFileSync(new URL('../configs/copy.json', import.meta.url)));
const source = `@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read_write> b:array<f32>;
const WIDTH = 128u * 2u;
@compute @workgroup_size(WIDTH) fn main(@builtin(global_invocation_id) id:vec3u) { if(id.x<arrayLength(&a)){b[id.x]=a[id.x];} }`;
test('parser resolves constant expressions, sparse bindings, selected entry resources', () => {
  const r = reflect(source + '\n@group(0) @binding(9) var<storage,read> unused:array<f32>;', 'main');
  assert.deepEqual(r.workgroup, [256,1,1]); assert.deepEqual(r.resources.map(r=>r.binding), [0,2]);
  for (const N of [1,16,255,256,257,1000]) { const c=copy(); c.parameters.N=N; assert.deepEqual(prepare(c,r).dispatch,[Math.ceil(N/256),1,1]); }
});
test('configuration rejects missing, extra, duplicate and incorrect access bindings', () => {
  const r=reflect(source,'main');
  for (const mutate of [c=>c.resources.pop(), c=>c.resources.push({...c.resources[0],binding:1}), c=>c.resources.push(c.resources[0]), c=>c.resources[0].access='read_write']) {
    const c=copy(); mutate(c); assert.throws(()=>prepare(c,r));
  }
});
test('explicit dispatch independent of N; 2D and unresolved elementwise rejected', () => {
  const c=copy(); c.dispatch={mode:'explicit',workgroups:[2,3,4]}; assert.deepEqual(dispatchFor(c,{workgroup:[8,8,1]}),[2,3,4]);
  c.dispatch.mode='elementwise'; assert.throws(()=>dispatchFor(c,{workgroup:[8,8,1]}),/only 1D/);
  assert.throws(()=>dispatchFor(c,{workgroup:null}),/statically resolved/);
});
test('group norm independent hand-computed cases across batches/groups/spatial axes', () => {
  const p={batch:2,channels:4,spatial:[1,2],groups:2,epsilon:0.25};
  const input=Float32Array.from([1,2,3,4,5,5,5,5,-4,-2,0,2,2,4,6,8]);
  const r=groupNorm(input,p);
  assert.deepEqual(r.means,[2.5,5,-1,5]); assert.deepEqual(r.variances,[1.25,0,5,5]);
  const expected=[-1.5,-.5,.5,1.5].map(x=>x/Math.sqrt(1.5)).concat([0,0,0,0],[-3,-1,1,3].map(x=>x/Math.sqrt(5.25)),[-3,-1,1,3].map(x=>x/Math.sqrt(5.25)));
  assert.ok(compare(r.output,expected,1e-14,0).passed);
  for (const bad of [{batch:0},{channels:3},{spatial:[0]},{groups:0},{epsilon:NaN},{epsilon:-1}]) assert.throws(()=>parameters({...p,...bad}));
  assert.equal(parameters(p).N,16);
});
test('checks report mismatching indices and nonfinite values; tolerances are asymmetric to reference', () => {
  const r=compare([1,NaN,Infinity,5],[1,2,3,4]); assert.equal(r.count,3); assert.equal(r.nonfinite,2); assert.deepEqual(r.mismatches.map(m=>m.index),[1,2,3]);
  assert.ok(compare([10.1],[10],0.01,0.01).passed); assert.equal(compare([10.2],[10],0.01,0.01).passed,false);
});
test('uniform field mapping tracks configuration and respects byte offsets', () => {
  const data=bufferData({group:0,binding:4,type:'u32',length:4,init:'fields',fields:[{offset:0,type:'u32',value:'channels'},{offset:8,type:'f32',value:'epsilon'}]}, {channels:8,epsilon:0.125});
  const v=new DataView(data.buffer); assert.equal(v.getUint32(0,true),8); assert.equal(v.getFloat32(8,true),0.125); assert.equal(v.getUint32(4,true),0);
});
test('group preset runs without confirmation and supports disabled intermediate checks', () => {
  const c=JSON.parse(fs.readFileSync(new URL('../configs/group-norm.json',import.meta.url)));
  assert.equal(prepare(c,reflect(source,'main')).p.N,64);
  c.check.mode='disabled'; assert.equal(prepare(c,reflect(source,'main')).p.N,64);
});
test('named spatial dimensions drive sequential inputs, output sizes and uniform packing together', () => {
  for (const size of [3,8,10]) {
    const c=JSON.parse(fs.readFileSync(new URL(`../configs/group-norm-uniform-${size}.json`,import.meta.url)));
    for (const groupSize of [size,5]) {
      c.parameters.group_size=groupSize;
      c.parameters.spatial=['group_size'];
      c.resources[0].init='sequential';
      const p=parameters(c.parameters);
      assert.deepEqual(p.spatial,[groupSize]);assert.equal(p.N,2*groupSize);
      const input=bufferData(c.resources[0],p),uniform=bufferData(c.resources[1],p),output=bufferData(c.resources[2],p);
      assert.deepEqual(Array.from(input),Array.from({length:p.N},(_,i)=>i+1));
      assert.equal(output.length,p.N);assert.ok(output.every(v=>v===0));
      const view=new DataView(uniform.buffer);assert.equal(uniform.byteLength,16);
      assert.equal(view.getUint32(0,true),groupSize);assert.equal(view.getFloat32(4,true),Math.fround(p.epsilon));
      const ref=groupNorm(input,c.parameters);
      assert.deepEqual(ref.means,[(groupSize+1)/2,(3*groupSize+1)/2]);
      assert.ok(ref.variances.every(v=>Math.abs(v-(groupSize**2-1)/12)<1e-12));
    }
  }
  assert.throws(()=>parameters({batch:1,channels:2,groups:2,epsilon:1e-6,spatial:['missing']}),/spatial dimension/);
});
test('ceiling channel partitions reset at each batch and normalize 16/16/8 elements', () => {
  const c=JSON.parse(fs.readFileSync(new URL('../configs/group-norm-ceil-80.json',import.meta.url)));
  const p=parameters(c.parameters), input=bufferData(c.resources[0],p), r=groupNorm(input,c.parameters);
  assert.equal(p.N,80);assert.deepEqual(p.spatial,[2,2]);
  assert.deepEqual(r.means,[8.5,24.5,36.5,48.5,64.5,76.5]);
  assert.deepEqual(r.variances,[21.25,21.25,5.25,21.25,21.25,5.25]);
  const spans=[[0,16],[16,32],[32,40],[40,56],[56,72],[72,80]];
  spans.forEach(([start,end],g)=>{
    for(let i=start;i<end;i++) assert.ok(Math.abs(r.output[i]-(i+1-r.means[g])/Math.sqrt(r.variances[g]+1e-6))<1e-12);
  });
  assert.ok(compare(groupNorm(new Float32Array(80).fill(7),c.parameters).output,new Float32Array(80)).passed);
  assert.throws(()=>parameters({...c.parameters,partition:'equal'}),/divisible/);
  assert.throws(()=>parameters({...c.parameters,channels:5,groups:4}),/empty group/);
  const uniform=bufferData(c.resources[1],p),view=new DataView(uniform.buffer);
  assert.equal(uniform.byteLength,32);
  [2,2,10,2,3].forEach((value,i)=>assert.equal(view.getUint32(i*4,true),value));
  assert.equal(view.getFloat32(20,true),Math.fround(1e-6));
  assert.ok(new Uint8Array(uniform.buffer,24).every(v=>v===0));
  assert.deepEqual(c.dispatch.workgroups,[6,1,1]);assert.equal(c.constants.WORKGROUP_SIZE,4);
});
