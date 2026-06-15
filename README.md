# WGSL Playground

## Running it

You need Chrome 113+ (or Edge). Firefox does not support WebGPU.

You also need to serve it over HTTP — WebGPU is blocked on `file://` URLs.
The easiest way:

```bash
# Python (built-in)
python3 -m http.server 8080

# Node
npx serve .

# VS Code: install the "Live Server" extension, right-click index.html → Open with Live Server
```

Then open `http://localhost:8080`.

---

## Writing kernels

### Option A: inline in the HTML (simplest)

Edit the `<textarea>` directly in the browser. Good for quick experiments.
No reload needed — just hit Run.

### Option B: separate `.wgsl` files (recommended)

Create a file next to `index.html`, e.g. `group_norm.wgsl`.
Then in `index.html`, replace the textarea default value with a fetch:

```js
// near the top of the <script> block, after init():
fetch('group_norm.wgsl')
  .then(r => r.text())
  .then(src => document.getElementById('editor').value = src);
```

Now you edit `group_norm.wgsl` in your editor of choice, reload the page, hit Run.
The textarea still shows the source so you can see it.

If you want live reload on file save, use the VS Code Live Server extension
or `browser-sync`:

```bash
npx browser-sync start --server --files "*.wgsl, index.html"
```

---

## Adding a new kernel

Every kernel needs two things: a JS harness (buffers + dispatch) and the WGSL shader.

### 1. JS side — in `index.html`

Create the input buffers and write data into them:

```js
const gammaBuf = device.createBuffer({
  size: C * 4,                                    // C floats, 4 bytes each
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(gammaBuf, 0, gammaData); // gammaData is a Float32Array
```

Add them to the bind group:

```js
{ binding: 2, resource: { buffer: gammaBuf } },
{ binding: 3, resource: { buffer: betaBuf  } },
```

Dispatch with the right workgroup count:

```js
pass.dispatchWorkgroups(numGroups); // one workgroup per group, typically
```

### 2. WGSL side — in your `.wgsl` file

Declare the same bindings:

```wgsl
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(3) var<storage, read>       beta:   array<f32>;
```

Use workgroup shared memory for reductions (mean, variance):

```wgsl
var<workgroup> shared: array<f32, 256>; // must be a compile-time constant size

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) { ... }
```

Synchronize threads within a workgroup after writing to shared memory:

```wgsl
workgroupBarrier();
```

---

## Tensor layout convention used here

All tensors are flat `array<f32>` in row-major order.

```
input[b, c, i] = input[b * C * L + c * L + i]
```

Where:
- `B` = batch size
- `C` = number of channels  
- `G` = number of groups
- `L` = spatial length (channels per group = C / G)

GroupNorm groups consecutive channels: group `g` contains channels `[g*L .. (g+1)*L)`.

---

## Debugging tips

- **Compile errors** appear in the error box with line numbers.
- **Wrong output** but no error: add `output[0] = f32(some_value)` to probe intermediate values.
- **All zeros**: usually a binding mismatch — check that JS binding indices match WGSL `@binding(N)`.
- **NaN/Inf**: usually a divide-by-zero — add an epsilon: `1.0 / sqrt(variance + 1e-5)`.
- The workgroup size in WGSL (`@workgroup_size(N)`) must match your dispatch math in JS (`dispatchWorkgroups(ceil(total / N))`).