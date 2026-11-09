# GPU improvements
Details for the GPU line of [improvements.md](improvements.md), one short line each, measurements go into commit messages. Diffusion models (Moss, H3) live in `../StrictComfy/`. Remove what is done, bugs go into improvements.md.

- A called method with a default parameter keeps the loop on the CPU (`fits` of MlirGpuCheck.IsCall wants every argument, MlirDeviceCalls only passes those of the call): this is what blocks `.Round` (its `decimals = 0`), the build fails with GpuLoopMustRunOnGpu.
- small number types as kernel values: values, parameters and members of types with 2-4 Number members (Complex) and lists of them read by index work (StructLanes, LaneCalls, MlirDeviceRecords), still on the CPU: a `for` over a List(Complex) (with a `value.Bits is 32` member it builds without GpuLoopMustRunOnGpu), a List(Complex) result, a method returning a Complex that has a loop or an early return, the 32 bit types (Vector2, Vector3, Size); a flattened list goes up again for every launch; a `to Text` reached only through text conversion is not emitted unless something calls it (why lane inlining skips `to`)
- convolutions of 64 bit pixels from a CPU loop miss strict_gpu_convolution without STRICT_GPU=all (H3 VAE encode of a PNG frame 12.5 instead of 0.7 s)
- Joining lists of different formats natively goes through the host (strict_gpu_concat needs equal formats, WebGPU has strict_joined): convert on the device.
- no native test reaches the locked path of `strict_packed_rows` and `strict_rows` (matrices of 4 MB and more): a test with a big matrix
