
override WORKGROUP_SIZE: u32 = 4u; // num of threads in a workgroup

// Tenth milestone: Parameterize params.epsilong and group size
struct Params {
    group_size: u32,
    epsilon: f32,
};

// Input buffer: TODO change dtype
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
// Scratchpad for partial sums, variances, etc
var<workgroup> partials: array<f32, WORKGROUP_SIZE>;
// Mean 
var<workgroup> mean: f32;
var<workgroup> variance: f32;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
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
    // Third milestone: parameterize with WORKGROUP_SIZE and params.group_size
    // Eight milestone: allow workgroup sizes that don't exactly divide
    // the input size
    var partial_sum : f32 = 0.0f;
    let starting_offset = u32(params.group_size * workgroup_id.x + lane);
    for (var i = starting_offset; i < params.group_size * (workgroup_id.x + 1u); i += WORKGROUP_SIZE) {
        partial_sum += input[i];
    }
    partials[lane] = partial_sum;

    //-------------------------------------
    // barrier for protecting partials
    workgroupBarrier();
    //-------------------------------------

    // Fourth milestone: Have thread 0 in each workgroup sum up the partials
    // and save the mean.
    if (lane == 0) {
        var total_sum : f32 = 0.0f;
        for (var i: u32 = 0u; i < WORKGROUP_SIZE; i++) {
            total_sum += partials[i];
        }
        mean = total_sum / f32(params.group_size);
    }

    //-------------------------------------
    // barrier for protecting mean
    workgroupBarrier();
    //-------------------------------------

    // Fifth milestone: Each thread sums (x - mean)^2 over same two elements it owns,
    // and writes that partial to debug output.
    var partial_sq_dev : f32 = 0.0f;
    for (var i = starting_offset; i < params.group_size * (workgroup_id.x + 1u); i += WORKGROUP_SIZE) {
        partial_sq_dev += (input[i] - mean) * (input[i] - mean);
    }
    partials[lane] = partial_sq_dev;

    //-------------------------------------
    // barrier for protecting partials
    workgroupBarrier();
    //------------------------------------- 

    // Sixth milestone: Reduce into variance by adding all sq dev in partials,
    // and dividing by group size.
    if (lane == 0) {
        var total_sum : f32 = 0.0f;
        for (var i = 0u; i < WORKGROUP_SIZE; i++) {
            total_sum += partials[i];
        }
        variance = total_sum / f32(params.group_size);
    } 

    //-------------------------------------
    // barrier for protecting variance 
    workgroupBarrier();
    //-------------------------------------

    // Seventh milestone: Write final group norm as y = (x-mean)/√(variance+ϵ).
    var group_norm : f32 = 0.0f;
    for (var i = starting_offset; i < params.group_size * (workgroup_id.x + 1u); i += WORKGROUP_SIZE) {
        let x = input[i];
        output[i] = (x - mean) / (sqrt(variance + params.epsilon));
    }


}
