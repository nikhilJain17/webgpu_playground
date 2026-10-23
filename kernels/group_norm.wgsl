// Input buffer: TODO change dtype
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

const WORKGROUP_SIZE = 4u; // num of threads in a workgroup
const GROUP_SIZE = 8u; // num of data in a group_norm's group

var<workgroup> partials: array<f32, WORKGROUP_SIZE>;
// TODO: paramterize workgroup size
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // First milestone: copy this group's elements.
    // Second milestone: calculate partial sums per workgroup.
    // Each thread should:
        // 1. Calculate its group’s starting offset.
        // 2. Read its two assigned elements.
        // 3. Add them.
        // 4. Write its partial sum into a unique output slot.
    // Third milestone: parameterize with WORKGROUP_SIZE and GROUP_SIZE
    if (gid.x >= arrayLength(&input)) { return; }
    let starting_offset = u32(GROUP_SIZE * group.x);
    var sum : f32 = 0.0f;
    for (var i = 0u; i < GROUP_SIZE / WORKGROUP_SIZE; i++) {
        sum += input[starting_offset + lane + (i * WORKGROUP_SIZE)];
    }
    partials[lane] = sum;
    output[gid.x] = sum;
}