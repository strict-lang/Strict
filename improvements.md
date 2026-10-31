# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- `strict test Compiler/NativeBuild.strict native`: the compiler stops with IndexOutOfRange in MlirMembered.Slot (MlirEmbedded.IsCopied) emitting the test module.
- `Compiler/check_gpu_cache.py`: mlir-opt rejects its cache-kind.mlir.
- Routed attention (routing > 0) with strided values (valueOffset/valueStride) runs unrouted: `strict_gpu_attention` takes the routed 8 bit path only for packed values, so H3Block.Heads with V read from the combined list attends densely (6 s slower, 12.9 dB from the routed result), routing 0 is bit-identical; the per-token value quantization (`strict_device_quantize_rows_run`) and the pooled value means need strided rows, H3 keeps the copy until then.
- `strict test NeuralNetworks native`: Safetensors.Tuned fails (3 tests); Speculation's Truncated (`Kept.Step(2, 4)`, line 88) fails only in the package run since the 1M GPU minimum, alone it passes with and without the GPU: it compares a run whose loops partly went to the GPU (lists earlier tests left there) with one that did not.
- A full heap while worker threads allocate (a parallel loop making texts) ends with OutOfMemory natively: the collector cannot run then (strict_gc_run defers while the pool is busy), seen with STRICT_HEAP=128.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.
- Networking: `Http.Send`, `Http.Started`/`Received` and `Hmac.Sha256` are in (Windows, Linux and the web checked, macOS untested), DeltaEngine's Google sign-in and sheet downloads use them: switch Rebalancing's `NativeIO.Http`/`Hmac` (with the user, it trades live); client certificates when a program needs them, TLS for multiplayer sockets later.
- GPU sidecars: every program still keeps Name.kernels.mlir (the PTX cache key) and test executables their PTX next to the source, move them into Tools/PtxCache once it evicts old entries (up to 25 MB each, never cleaned today).
- Realtime frames: a full collection at a frame end that takes longer than a few ms marks over several frame ends (incremental, the cards record what changed meanwhile).
- f32 kernels: a GPU loop reading a `value.Bits is 32` member computes in f32 (WGSL on WebGPU, the 32 bit CUDA kernel natively); WGSL lacks Sin, Cos, Logarithm, `from` calls, Bytes results and rows.

## Waiting
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

