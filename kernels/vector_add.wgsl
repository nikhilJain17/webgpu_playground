// Declare input and output arrays
@group(0) @binding(0) var<storage, read>       input1:  array<f32>;
@group(0) @binding(1) var<storage, read>       input2:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Simply double each element
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&input1)) { return; }
  output[i] = input1[i] + input2[i];
}