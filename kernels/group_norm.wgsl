
// Declare input and output arrays
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
// @group(0) @binding(1) var<storage, read>       input2:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// declare shared_mem memory same as workgroup size
// each thread gets an array slot
var<workgroup> shared_mem: array<f32, 64>;
var<workgroup> sum: f32;
var<workgroup> mean: f32;
var<workgroup> shared_sq_dev: array<f32, 64>;
var<workgroup> variance: f32;

// Simply double each element
@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) local_index: u32
) {
    // 1. Calculate parallel sum
    // Initialize shared_mem memory
    if (local_index < arrayLength(&input)) {
        shared_mem[local_index] = input[local_index];
    }
    workgroupBarrier();
    // Reduce array by half every time
    for (var i: u32 = 0u; i < u32(log2(f32(arrayLength(&input)))); i++) {
        // Add even-numbered indices with their neighbors
        // and store into index/2 every iteration,
        // dividing the size of the task by 2.
        var local: f32 = 0;
        var write_idx: u32 = local_index;
        // Do all the reads
        if (local_index < arrayLength(&input) / u32(pow(2f,f32(i))) && local_index % 2 == 0) {
            // read
            local = shared_mem[local_index] + shared_mem[local_index + 1];
        }
        // Barrier
        workgroupBarrier();
        // Do all the writes
        if (local_index < arrayLength(&input) / u32(pow(2f,f32(i))) && local_index % 2 == 0) {
            // write
            write_idx = local_index / 2;
            shared_mem[write_idx] = local;
        }
        // Barrier
        workgroupBarrier();
    }
    workgroupBarrier();
    
    // 2. Calculate parallel mean from parallel sum
    // TODO: Should every thread do this division?
    // or one thread do it and everyone waits for it to be done?
    sum = shared_mem[0];
    mean = sum / f32(arrayLength(&input));
    shared_mem[1] = mean; 
    
    // 3. Calculate parallel variance from parallel sum
    if (local_index < arrayLength(&input)) {
        shared_sq_dev[local_index] = pow(input[local_index] - mean, 2);
    }
    // parallel sum again
        for (var i: u32 = 0u; i < u32(log2(f32(arrayLength(&input)))); i++) {
        // Add even-numbered indices with their neighbors
        // and store into index/2 every iteration,
        // dividing the size of the task by 2.
        var local: f32 = 0;
        var write_idx: u32 = local_index;
        // Do all the reads
        if (local_index < arrayLength(&input) / u32(pow(2f,f32(i))) && local_index % 2 == 0) {
            // read
            local = shared_sq_dev[local_index] + shared_sq_dev[local_index + 1];
        }
        // Barrier
        workgroupBarrier();
        // Do all the writes
        if (local_index < arrayLength(&input) / u32(pow(2f,f32(i))) && local_index % 2 == 0) {
            // write
            write_idx = local_index / 2;
            shared_sq_dev[write_idx] = local;
        }
        // Barrier
        workgroupBarrier();
    }
    workgroupBarrier();
    // TODO: Should every thread do this division?
    // or one thread do it and everyone waits for it to be done?
    variance = shared_sq_dev[0] / f32(arrayLength(&input)); 

    // 4. Calculate element-wise formula
    if (local_index < arrayLength(&input)) {
        output[local_index] = (input[local_index] - mean) / pow(variance - 0.0005, 0.5);
    }

}