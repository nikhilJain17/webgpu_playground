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

# Node with auto-reload on file changes
npx live-server --port=8080

# VS Code: install the "Live Server" extension, right-click index.html → Open with Live Server
```

Then open `http://localhost:8080`.

---

## Writing kernels

Write your WGSL shader in your editor, then paste it into the textarea and hit **Run**.
Compile errors appear in the error box with line numbers.

---

## Validating WGSL offline with `tint`

`tint` is the WGSL compiler used inside Chrome. Running it locally catches errors before opening the browser.

**Install** (macOS via Homebrew):

```bash
brew install tint
```

Or build from source: https://dawn.googlesource.com/dawn

**Usage:**

```bash
tint validate kernels/group_norm.wgsl
```

Exit code 0 = valid. Errors include line numbers and a message from the same parser Chrome uses.

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
