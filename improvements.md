# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Ternary Bonsai 27B at a simulated 10 GB stalled once more (first run after a cold build), 12 later runs did not: [improvements-models.md](improvements-models.md).
- H3 image to video with `--image` and `--last_image` puts only the first image into the Qwen3-VL context (`Qwen3Images.Prefilled` takes one image, ComfyUI tokenizes both).
- MossImage (StrictComfy, best, 2 images, seed 7) gives another first image in 2 of 12 runs on a quiet machine (39.7 and 44.5 dB from the usual one, after a cold load StrictComfy saw 35.4 dB), the second image stays identical unless another process holds 1 GB of the GPU (then it differs too, 45.6 dB, with sampling 7.1 instead of 3.1 s from streaming): numbers depend on residency or timing, the kernel mix is the same (42 `strict_device_wholes` in every first image), no float atomics; 6 runs with STRICT_GRAPHS=0 were all identical.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

