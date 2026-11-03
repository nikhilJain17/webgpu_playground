const D: u32 = 6u;
const H: u32 = D / 2u;
const MAX_PERIOD: f32 = 1000.0;

@group(0) @binding(0)
var<storage, read> timesteps: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let global_thread_index = gid.x;
    let num_timesteps = arrayLength(&timesteps);

    // TODO: Guard extra invocations, calculate n and j, and write the embedding.
    if (global_thread_index >= num_timesteps * H) { return; }

    let timestep_index: u32 = global_thread_index / H;
    let frequency_index: u32 = global_thread_index % H;
    
    let frequency: f32 = exp(-log(MAX_PERIOD) * f32(frequency_index) / f32(H));
    let angle: f32 = timesteps[timestep_index] * frequency;

    output[timestep_index * D + frequency_index] = cos(angle);
    output[timestep_index * D + frequency_index + H] = sin(angle);
}
