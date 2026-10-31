# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- `strict test Compiler/NativeBuild.strict native`: the compiler stops with IndexOutOfRange in MlirMembered.Slot (MlirEmbedded.IsCopied) emitting the test module.
- `Compiler/check_gpu_cache.py`: mlir-opt rejects its cache-kind.mlir.
- Routed attention (routing > 0) ignores valueOffset/valueStride of strided values (dd496fb0): H3Block.Heads with V read from the combined list gives 12.9 dB against the copy and is 6 s slower, routing 0 is bit-identical; `Attention.Strided` only tests routing 0, H3 keeps the copy.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.
- Networking: `Http.Send`, `Http.Started`/`Received` and `Hmac.Sha256` are in (Windows, Linux and the web checked, macOS untested), DeltaEngine's Google sign-in and sheet downloads use them: switch Rebalancing's `NativeIO.Http`/`Hmac` (with the user, it trades live); client certificates when a program needs them, TLS for multiplayer sockets later.
- GPU sidecars: every program still keeps Name.kernels.mlir (the PTX cache key) and test executables their PTX next to the source, move them into Tools/PtxCache once it evicts old entries (up to 25 MB each, never cleaned today).
- f32 kernels: a GPU loop reading a `value.Bits is 32` member computes in f32 (WGSL on WebGPU, the 32 bit CUDA kernel natively); WGSL lacks Sin, Cos, Logarithm, `from` calls, Bytes results and rows.

## Waiting
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

