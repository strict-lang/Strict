# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Generated WGSL loops compute whole numbers as 32 bit integers and overflow silently above 2^31 (native uses 64 bits): Qwen3Block.HeadScales multiplied 10.5 million by 256 and an index by 439808 at 480x848 and read wrong rows on WebGPU only (reordered in 14b1f46d); detect the overflow (fall back or fail the loop) instead of wrong numbers.
- Native Safetensors test `Nibbles.Tuned(...)` (line 21, in Entry) exceeds the 100 ms test budget every run (also before the rank above 512 GPU merge), passes on the VM.
- Routed attention (routing > 0) with strided values (valueOffset/valueStride) runs unrouted: `strict_gpu_attention` takes the routed 8 bit path only for packed values, so H3Block.Heads with V read from the combined list attends densely (6 s slower, 12.9 dB from the routed result), routing 0 is bit-identical; the per-token value quantization (`strict_device_quantize_rows_run`) and the pooled value means need strided rows, H3 keeps the copy until then.
- `"abc".Reverse` (a List method returning the list, called on a Text) gives `(", c, b, a, ")` on the VM and `undefined symbol: Text_Reverse` natively (Ben). `Text(text.characters.Reverse)` works on the VM, natively a Text's `characters` is the Text handle itself: `Length` works on it, Reverse, First and `Text.Combine` of it see an empty list. Decide one native List(Character) representation, then forward such calls to `characters` in ValueCodegen and convert back with `Text(...)`.
- Native CUDA: a GPU loop reading a device-resident result of a 64 bit loop over Float32s (`Transposed` of Moss.Head) stops with "a loop on the GPU read outside of a list" after an earlier call of the same loop with Numbers (FluxPreview.Pixels twice); reading the result on the host first avoids it, the VM and the web are fine.
- Native `to Text` of a Float32 fails to compile ("unsupported: text of Float32"), every type should get Any's `to Text`.
- `ImageCodec.SaveJpeg` only works on Windows and the web (`JpegRequiresWindowsOrWeb` on Linux, macOS, Android, iOS): a baseline JPEG encoder in Strict next to ImageSaver's PNG would cover every platform.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.
- Networking: `Http.Send`, `Http.Started`/`Received` and `Hmac.Sha256` are in (Windows, Linux and the web checked, macOS untested), DeltaEngine's Google sign-in and sheet downloads use them: switch Rebalancing's `NativeIO.Http`/`Hmac` (with the user, it trades live); client certificates when a program needs them, TLS for multiplayer sockets later.
- GPU sidecars: every program still keeps obj/Name.kernels.mlir (the PTX cache key) and its PTX in obj/ next to the source, move them into Tools/PtxCache once it evicts old entries (up to 25 MB each, never cleaned today).
- Realtime frames: a full collection at a frame end that takes longer than a few ms marks over several frame ends (incremental, the cards record what changed meanwhile).
- f32 kernels: a GPU loop reading a `value.Bits is 32` member computes in f32 (WGSL on WebGPU, the 32 bit CUDA kernel natively); WGSL lacks Sin, Cos, Logarithm, `from` calls, Bytes results and rows.
- MossImage in the browser (prompt in, encoder, Moss and VAE cached in OPFS, WebGPU like FractalZoomer): the VAE decode runs as WGSL kernels (Compiler/NativeRuntimeWgsl.js) with kept 16 bit maps; next Moss (an int8 GEMM with dot4I8Packed), the encoder, OPFS for the big models.
- Release builds: platform builds into Release/<Platform> are optimized without debug checks, logging or per-call checks (the web runtime already is), plain `strict build`, test executables and the VM keep them; native needs a release flag on `strict build` for local benchmarks.
- Native symbols, register kinds and helpers use the short type name, so one executable cannot hold two packages' types of the same name: qualify the symbols of a type whose short name is taken.
- Named iterators over counts (`for row, column in rows, width`) as one flat loop of rows * width with the names from the index (parallel, GPU kernel), then the `(index / n).Floor` and `index % n` loops of NeuralNetworks, Math/Matrix and the image decoders can name their iterators without losing their kernels.
- Parallel dedup by key (Ben): a `for` that keeps the first element per computed key (Dictionary or distinct by key) in parallel; Ben's program search keeps 2 of 128 cores busy because its observational equivalence dedup is a sequential loop over a mutable Dictionary.

## Waiting
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.
- Later for Ben's long-term memory: an embedded persistent key-value store (LMDB-like, memory mapped, hash keys to blobs) usable on the VM and natively, once skill and evidence files get too many.

