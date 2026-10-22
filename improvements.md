# Improvements
In priority order, short and actionable. Remove what is done (results go into the README of the package), add what is found. Diffusion models (Moss, Flux, MiniMax) and GPU speed are worked on from `../StrictComfy/` (plan in `../StrictComfy/Plan.md`).

## Next
1. **CI complete and fast**: `.github/workflows/ci.yml` runs the bootstrap, one self build and the VM tests of the fast folders (Linux about 5 minutes with `Examples`, Windows only the quickest folders, 10 at most). Add `Validators`, `Runtime`, `Expressions` and native builds back while it stays under 10 minutes; a failing native build prints the file and exit code. CI, `Toolchain.LinuxRelease`, `bootstrap.sh` and `MacOSTools` still use LLVM 22, move them to 23.1.3 like Windows (msys2 LLVM 23 needs cc-libs and gcc-libs 16, clang 23 too: LLVM 23 writes float literals as `f0x...`, which clang 22 can not read).
2. **Faster builds**: a gated program builds twice (`Name.test.exe`, then `Name.exe`, each compiling the bytecode from source), one bytecode load for both would save the second generation; native gates write no `.strictbinary`; generation is single threaded (10.8 s of `MlirModule.Declared` for `strict.exe`); the host build waits for the PTX; `--mlir-disable-threading` was needed for a threading race in 21.1.7, with 23.1.3 threaded runs give identical output (mlir-opt of `strict.exe` 4.2-4.7 -> 3.55 s, mlir-translate 2.5 -> 2.3 s): drop it once every platform uses 23.
3. **16 bit lists on the GPU** (in progress): a list of at least 2^20 numbers computed on the GPU is stored as bfloat16 (format 17: the high half of a float, same range, 8 bits of precision; Moss residuals reach 41472, f16 could overflow), smaller ones (norm scales, modulations, logits, partial sums) stay 32 bit floats, 16 bits would save nothing there. Arithmetic stays in 32 bit registers: every list read converts (strict_device_elements, host unpacking), generated kernels write the format strict_gpu_run passes as %results, their guarded body reads 16, 17 and 32 bit lists, the int8 products epilogue, attention, quantize, hadamard, products and convolution read and write format 17. Lists joined with another format go through the host.
4. **Qwen3.8 for the robot arm**: speech to text, a small Strict endpoint the arm calls, the time from question to first sound.

## Bugs
- Qwen3.8 prompts of about 920 tokens and more stop with `a loop on the GPU read outside of a list`: refreshing the f32 value from the exact parts after every operation in 32 bit kernels (d0f4a743) let 930 tokens prefill, but changed Qwen3.8's tokens and cost 4 ms per decoded token, so it was taken out; refreshing only whole numbers past 2^24 did not fix it.
- `strict check` does not report an argument of the wrong type for a constructor (`MlirLines(value)` inside a `for` over instructions built an `MlirLines` from an `InstructionEntry`, the VM ran it, natively it crashed).

## Later
Language model research (`NeuralNetworks/README.md`), big models and memory (Qwen3-Coder-30B with all experts), image blur on the GPU, web and mobile platforms, Apple Silicon, AMD GPUs.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Qwen3-Coder-Next 80B-A3B pruned to 18-20 GB: needs more VRAM and the `qwen3next` architecture.
