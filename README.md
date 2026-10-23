# WGSL Playground

A small WebGPU harness for developing buffer-based compute shaders. Edit WGSL and configuration, without editing JavaScript for each kernel. Shader files in `kernels/` are your exercises; the harness does not modify them.

## Run

Install the pinned dependencies once, then serve this directory:

```bash
npm ci
npm start
```

Open **http://localhost:8080** in a WebGPU-enabled browser. The page reports whether a device is available. `python3 -m http.server 8080` is equivalent to `npm start`; there is no build step. Use localhost or HTTPS, not `file://`. Dependencies are served locally from `node_modules`, with no CDN required at runtime.

## Copy workflow

1. Open your WGSL file, load an existing file, or paste into the source editor.
2. Select **Copy** and click **Apply preset**. This configures F32 input at group 0 / binding 0 and F32 output at group 0 / binding 2. There is **no binding 1**. Change the resource rows if your shader uses other bindings.
3. Set `N` in Parameters. Both buffers use length `N`. Input is deterministic: `((i * 17 % 101) - 50) / 8`, exactly representable in F32.
4. Select **Elementwise** and run. A shader with `@workgroup_size(256)` dispatches `ceil(N / 256)` workgroups. The shader must guard indices beyond the input length.
5. Read **Execution** and **Correctness** separately. Copy comparison is exact numeric F32 equality (zero tolerances). NaN and Infinity always fail.

The initial file is `kernels/hello_world.wgsl`, which currently doubles input. Initial checking is therefore **disabled**; applying Copy explicitly enables the copy check. Loading a shader never selects a preset based on its filename.

## Four independent pieces

- **Source:** the editor is the sole source executed. Disk files are fetched on load, not copied into JavaScript. The source label identifies the loaded file and unsaved changes; each run records its source identity. Download editor saves a separate file and does not claim to overwrite the original. Reloading with edits asks before discarding them. Download your edits before leaving.
- **Resources:** explicit group, binding, storage/uniform kind, access, role, scalar representation, length, initializer and readback choice. No output role is inferred from binding order. Add extra `debug` buffers and enable readback to inspect intermediate values.
- **Dispatch:** Elementwise derives coverage from `N` and the reflected workgroup size; Explicit uses the X/Y/Z workgroup counts you supply, independently of `N`.
- **Checks:** Disabled, Copy, or Group normalization. Input/output are chosen by `group:binding`. Disabling checks still performs compilation, pipeline validation, execution and requested readbacks.

The reusable configuration files are [`configs/copy.json`](configs/copy.json) and [`configs/group-norm.json`](configs/group-norm.json). Import/export JSON configurations in the UI. Applying a preset replaces configuration, never shader source.

### Resource reflection and layout

[`wgsl_reflect` 1.6.0](https://github.com/brendan-duncan/wgsl_reflect) parses WGSL and reflects the selected compute entry point’s resources and workgroup dimensions. **Inspect shader resources** lists the required declarations, minimum sizes, array strides and struct member offsets. No regex-based WGSL parser is used.

Only resources statically used by the selected entry point belong in its configuration, including resources used through called functions. Missing, extra, duplicate or incompatible bindings fail explicitly. The runner creates an automatic pipeline layout and obtains each actual bind-group layout from that pipeline. Browser validation remains authoritative. Parser limitations are reported as reflection failures, even for browser-valid WGSL; no source is silently rewritten. For example, this version accepts a named constant for workgroup size but rejects a direct attribute expression such as `@workgroup_size(W * 2)`; define that expression in a `const` and use its name. Explicit dispatch does not bypass parsing. Unused shader declarations do not cause dummy buffer creation.

Storage access supports `read` and `read_write`; uniform buffers use `read`. Roles describe your intent and do not change WGSL access. Buffer contents and readbacks support F32, U32 and I32 scalar views. Length is a positive scalar count or a parameter name such as `N`; it is not bytes. All initializers operate on that scalar-sized allocation:

- `deterministic`: the copy input pattern above (intended for F32).
- `zero`: explicitly zero-initialized data, useful for outputs and scratch buffers.
- `values`: enter `{"values":[1,2,3,4]}` in the row’s JSON editor; count must match length.
- `fields`: pack fields at explicit byte offsets. Values may be numeric literals or parameter names. This supports mixed uniform structs and explicit padding.

For example, a **16-byte** uniform allocation uses type `u32`, length `4`, init `fields`. If your shader’s declared layout places channels at byte 0 and epsilon at byte 8, enter:

```json
{"fields":[
  {"offset":0,"type":"u32","value":"channels"},
  {"offset":8,"type":"f32","value":"epsilon"}
]}
```

This binds current parameter values on every run. Match offsets/types to your shader’s reflected layout; the harness does not invent a uniform declaration. Structs, vectors and matrices can be packed explicitly, but automatic serialization of arbitrary composite types is not provided. Padding is initialized to zero. Buffer size is limited to 16M scalars per resource and the device’s limits.

### Dispatch boundaries

**Elementwise** currently requires a statically resolved one-dimensional workgroup (`Y=Z=1`). Constant expressions supported by the parser are resolved. Multi-dimensional or unresolved workgroups produce an explanation directing you to Explicit mode. For conservative handling, any entry point that uses pipeline overrides requires Explicit mode; pass override values using Advanced pipeline constants JSON.

**Explicit workgroup count** supports X/Y/Z and checks the device dispatch limit. Counts must be positive integers. It is appropriate for reductions/group norm, where a workgroup may process many elements. The harness does not assume one workgroup per group, workgroup size equal to data size, or any algorithmic meaning for a filename.

## Group-normalization exercise

Apply the Group normalization preset separately from loading your shader. Its parameters are:

```json
{"batch":1,"channels":4,"spatial":[4,4],"groups":1,"epsilon":0.00001}
```

Dimensions and group count must be positive integers, channels must be divisible by groups, and epsilon must be positive and finite. `N = batch * channels * product(spatial)` is derived; a supplied `N` does not override this. The preset starts with an explicit `[1,1,1]` dispatch. **Choose counts appropriate to your implementation** when changing dimensions.

The CPU reference uses contiguous channels, flattened spatial dimensions and row-major indexing:

```text
S = product(spatial)
index(b, c, s) = (b * channels + c) * S + s
channels in group g = [g * channels/groups, (g+1) * channels/groups)
normalized = (x - mean) / sqrt(population_variance + epsilon)
```

It uses an independent two-pass mean/variance calculation in JavaScript double precision over the actual F32 input. There is no affine scale or bias. Reference means and variances are displayed as well. GPU output passes when every finite value satisfies:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

Default tolerances are `atol=1e-5`, `rtol=1e-4`; both are configurable. Failures show total mismatches, nonfinite pairs and the first 32 mismatching indices/values. Every readback separately reports nonfinite values even when checking is disabled.

The CPU reference uses the configured dimensions, grouping and epsilon. For parameterized shaders, use `fields` mappings to bind values from the same configuration. Hardcoded shader parameters remain under your control; mismatches are reported by the comparison. No confirmation checkbox is required.

While developing partial sums, means or variances, select **Disabled** and read back whichever output/debug buffers your shader declares. Neither the preset nor the tests supply or complete your group-norm kernel. The existing exercise is not asserted to implement the general reference contract.

## Errors and resource lifetime

Runs are serialized and inputs are frozen during execution. Each run starts with cleared results and owns its buffers, which are destroyed in `finally`, including on failure. The device is reused. Device loss is reported and the next run can request a new device.

Diagnostics include compiler source lines/columns and stage labels for reflection, configuration/bindings, pipeline validation, buffer/binding validation, command submission and readback. GPU stages use validation, out-of-memory and internal error scopes. An uncaptured-error listener is a fallback and invalidates displayed results. A device-loss handler also invalidates results. A fatal uncaptured error requires page reload; it is not silently cleared.

Failed execution never publishes readbacks or a numerical pass. Successful submission/readback and numerical correctness have separate labels; **execution success with checking disabled does not establish that a shader wrote correct values**. The UI previews the first 256 values of each readback; Download all readbacks exports every index, preserving nonfinite values as strings.

## Validation and tests

```bash
npm test
npm run test:browser
```

Browser tests launch installed Chrome, serve the project temporarily on `127.0.0.1:8093`, and run actual WebGPU. The default executable is macOS Google Chrome; set `CHROME_PATH` for another installation. The script currently selects the Metal ANGLE backend for this macOS environment. Reports/screenshots are written to ignored `test-results/`.

Verified on Chrome **155.0.8059.86**, Apple **metal-3** WebGPU adapter:

- Copy lengths **1, 16, 255, 256, 257, 1000**, each with workgroup sizes **32, 64, 128, 256**, using bindings **0/2** only.
- Syntax errors and recovery; missing/extra/access-mismatched bindings; actual GPU pipeline, bind-group and command validation errors.
- Debug-buffer readback, uniform packing, sparse groups/bindings, unused declarations and helper resources, explicit XYZ dispatch, overrides, and rejected 2D elementwise dispatch.
- Numerical failures and nonfinite values; overlap prevention; repeated runs with allocation/destruction accounting.
- Injected readback rejection with cleanup/recovery, actual device destruction/loss and recreation, uncaptured validation fallback; UI failure clears prior successful output.
- Independent CPU tests for reference values across batches/groups/spatial dimensions, invalid group-norm configuration, parameter packing and exact/tolerant comparison.

Browser fixtures only contain small harness probes; they do not modify `kernels/` or implement group norm.

### Optional offline Tint check

For the locally installed Dawn `tint` CLI:

```bash
tint --format wgsl kernels/group_norm.wgsl > /dev/null
```

There is no `validate` subcommand in this CLI. Exit code 0 means WGSL passed translation/validation, not that the algorithm is correct. `tint --help` describes your installed version. Tint is optional; the playground uses the browser’s compiler for execution diagnostics.

Your development loop:

```text
Edit WGSL
   ↓
Optional Tint: does it compile?
   ↓
Playground: do configuration, bindings and pipeline validate?
   ↓
Dispatch + readback: inspect intermediate values or compare against a reference
```

Implementation: `core.js` handles reflection/configuration/references, `runner.js` owns the WebGPU lifecycle, and `app.js` connects them to the editor. No generated or duplicate copies of exercise shaders are maintained.

## External group-norm kernel: inputs 1…16

Open this preconfigured page after starting the server:

[Group norm — external source, 16 elements](http://localhost:8080/?config=group-norm-16&source=external/stable-diffusion-group-norm.wgsl)

It loads `configs/group-norm-16.json` and the external shader at
`../stable-diffusion.cpp/ggml/src/ggml-webgpu/wgsl-shaders/group_norm.wgsl`.
The `external/stable-diffusion-group-norm.wgsl` filesystem symlink points to that
file; it is not a copied shader. The sibling `stable-diffusion.cpp` checkout must
exist at that relative location. The existing Python server follows this link.
**Load / reload file** fetches fresh source after edits in your shader editor.
Run executes the currently displayed editor snapshot, with its full source path,
unsaved-change status and SHA-256 shown under Last execution.

This preset provides:

- Binding 0:0: F32 inputs `[1,2,…,16]`, with readback enabled.
- Binding 0:2: **16** F32 output elements, with readback enabled.
- Explicit dispatch **2 × 1 × 1**, independent of the shader's four-thread workgroup.
- CPU group norm: batch 1, channels 2, spatial `[8]`, groups 2, epsilon `1e-6`.
- Full 16-element comparison: absolute and relative tolerances both `1e-5`.

The two reference groups have means **4.5** and **12.5**, population variance
**5.25** each, and the same expected eight normalized values. Click Run to compare all outputs. The preset can also be selected from the
configuration menu without changing source, or imported as JSON.

Reproduce the browser verification with:

```bash
npm run test:group-norm
```

This loads the page/preset, verifies the editor equals the requested external file,
checks the displayed source hash, compares all outputs over 20 actual GPU runs,
and deliberately breaks a binding to verify stale output is cleared. It never
edits the kernel. Numerical failures are recorded as **FAIL** in the report;
the test command verifies harness behavior rather than requiring an unfinished
kernel to pass. See `test-results/group-norm-16.json` and its screenshot.

On Chrome 155 / Apple Metal, source SHA-256
`d925e9c7cc4b167401e50cff8a0efe091a3b61b65edabf30989dc541c3bf428a`
produced 17 passing and 3 failing numerical runs out of 20. The failing runs had
index **4 = 5.25**, expected **0.21821786945333915**, with no nonfinite values.
The leftover `output[gid.x] = variance;` write remains in that source: workgroup 1
writes variance at output index 4 while workgroup 0 writes normalized output
there. Workgroup barriers cannot synchronize these separate workgroups. A pass
in one execution does not prove race freedom. The harness leaves this fix to you.
