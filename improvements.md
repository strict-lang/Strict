# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Routed attention (routing > 0) with strided values (valueOffset/valueStride) runs unrouted: `strict_gpu_attention` takes the routed 8 bit path only for packed values, so H3Block.Heads with V read from the combined list attends densely (6 s slower, 12.9 dB from the routed result), routing 0 is bit-identical; the per-token value quantization (`strict_device_quantize_rows_run`) and the pooled value means need strided rows, H3 keeps the copy until then.
- `strict test . native` (recursive, one build): Examples/Register and Examples/InstructionType share their names with the Bytecode types, but native symbols, register kinds and helpers use the short type name, so one executable cannot hold both ("unsupported: member Register.RZero in ExecuteOperation"); rename the two examples or qualify symbols of types whose short name is taken. Without them the build passes, in about 4 minutes (LLVM for 6234 functions).
- GPU state leaks between tests of one native folder executable: `strict test NeuralNetworks native` fails Safetensors.Tuned (lines 58-60) and Speculation's Truncated (line 88), StrictComfy's `strict test . native` FluxDecoder, FluxResnet, Moss.Ran and H3AudioBlock.Output (2.7 ms there instead of 314 ms alone); each file alone and STRICT_GPU=0 pass.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.
- Networking: `Http.Send`, `Http.Started`/`Received` and `Hmac.Sha256` are in (Windows, Linux and the web checked, macOS untested), DeltaEngine's Google sign-in and sheet downloads use them: switch Rebalancing's `NativeIO.Http`/`Hmac` (with the user, it trades live); client certificates when a program needs them, TLS for multiplayer sockets later.
- GPU sidecars: every program still keeps Name.kernels.mlir (the PTX cache key) and test executables their PTX next to the source, move them into Tools/PtxCache once it evicts old entries (up to 25 MB each, never cleaned today).
- Realtime frames: a full collection at a frame end that takes longer than a few ms marks over several frame ends (incremental, the cards record what changed meanwhile).
- f32 kernels: a GPU loop reading a `value.Bits is 32` member computes in f32 (WGSL on WebGPU, the 32 bit CUDA kernel natively); WGSL lacks Sin, Cos, Logarithm, `from` calls, Bytes results and rows.
- MossImage in the browser (prompt in, encoder, Moss and VAE cached in OPFS, WebGPU like FractalZoomer), VAE decode first: a web build of anything using Safetensors fails compiling NeuralNetworks/native/GgufFile.c (`ggml-cpu/quants.h` missing for wasm); convolution, GroupNorm, attention and the int8/16 bit products are CUDA runtime kernels without WGSL versions (FluxDecoder's web build gets one shaded loop), the CPU fallback decodes far too slowly; then weights streamed from OPFS into GPU buffers (4 GB wasm memory, WebGPU can't read host memory).

## Waiting
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

