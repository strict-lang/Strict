# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- A type level call from another type runs the method on the caller's instance (Invoke instance -1): `has items Texts = ("a", "b", "c")` with `Count Number` returning `items.Length`, `Bag.Count` called in another type's Run gives 1 on the VM (the caller's logger slot) and IndexOutOfRange natively for `items.First`, `DataFiles.Of` calling its own `Selected` crashed strict.exe (AccessViolation). Inside Bag's own tests it works. Either construct the default instance or reject the call when the method reads members.
- The GPU budget ignores other programs on Windows: WDDM's `cuMemGetInfo` reported 22.7 of 24 GB free while another process held 12.5 GB, so Strict grew to its full peak and Windows paged the other process's memory to RAM, slow tokens until done (this was the Ternary Bonsai "10 GB stall", on the first run under every fresh holder); next to ComfyUI the figure moved by 0.9 GB between starts. DXGI's QueryVideoMemoryInfo budget and its change notification give the real limit.
- A pure loop over GPU heavy calls forks onto CPU threads (MlirCost's threshold is 2 for a costly body): `Qwen3Images.Encoded` of 2 images ran its vision towers in workers and did not finish in 4.5 min, sequential takes 16 s more than one image.
- MossImage (StrictComfy, best, 2 images, seed 7) gives another first image in 2 of 12 runs on a quiet machine (39.7 and 44.5 dB from the usual one, after a cold load StrictComfy saw 35.4 dB), the second image stays identical unless another process holds 1 GB of the GPU (then it differs too, 45.6 dB, with sampling 7.1 instead of 3.1 s from streaming): numbers depend on residency or timing, the kernel mix is the same (42 `strict_device_wholes` in every first image), no float atomics; 6 runs with STRICT_GRAPHS=0 were all identical.
- GPT-2 got slower since its 2026-10-22 numbers on the 3080: `Gpt2Benchmark` decodes 1.74 instead of 1.10 ms per token on the GPU and 10.6 instead of 9.3 ms with STRICT_GPU=0 (BenchBlur on the GPU 7.8 instead of 6.6 ms, 5.5 ms of CPU before its 0.7 ms kernel); the Oct 20 Gpt2Text.exe still runs, older commits need their own bootstrap to bisect (the current compiler emits calls their runtime lacks, LLVM 22 runtimes do not build with 23)

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

