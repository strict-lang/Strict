# GPU improvements
Details for the GPU line of [improvements.md](improvements.md), one short line each, measurements go into commit messages. Diffusion models (Moss, H3) live in `../StrictComfy/`. Remove what is done, bugs go into improvements.md.

## Attention
- `strict_device_dense_attention` keeps the 8 bit tensor pipe 53 % busy (Moss on the 3080, 1.26 ms per layer, 253 registers leave one block per SM; comfy_kitchen's `int8_attention` was ~20 % faster on the 4090 before warps 4 to 7 ran half a step behind): fewer softmax instructions (part of the exponentials as a polynomial on the fma pipe), or two blocks per SM
- the row maximum on the integer sums (IMNMX) needs one key scale per lane and step instead of one per position
- the 2 tile copies of a routed step sit in conditional blocks (long scoreboard stalls after them): branch-free copies like `strict_device_dense_issued`
- the routing planner reads all pooled keys of a head in every block: moments once per head, the scores as a small product kernel
- the Hadamard on Q and K
- routing plans at most 1024 key blocks (65536 positions), longer prompts attend densely: plans in global memory
- head size 64 (H3 VAE decode) has no tensor core attention kernel

## 16 bit lists
- bf16 QK attention beyond the 128 wide QKV path
- attention, product and Hadamard results are 32 bit
- joining lists of different formats on the GPU goes through the host: convert on the device
- LLVM 23 PTX of strict_device_whole(s) is ~3 % slower than LLVM 22's
- compiled loops index in 64 bits: 32 bit index math where the counts fit

## Memory
- 4 bit rows (`Safetensors.Quantized(name, 4)`) keep the int8 row stride on the GPU (no VRAM saved yet), `Batch.Sums` multiplies them as 4 bit numbers (`Batch.Gated` goes the same way untested, `Batch.Grouped` and `GpuMemory.Tuned` must not get them: the GPU copy holds two numbers per byte), a host copy packed in place (kept 3) is no longer readable on the CPU
- streamed weights are not overlapped with compute (H3 idles ~230 ms per step, ComfyUI 60): uploads ahead in a ring need a cheaper refill than walking the chain
- without spilled rows the trim before an own block runs at every block
- full collections before big activation blocks (H3 ~98 per step, 18-28 ms each in strict_gc_mark_range): skip lists without references in the mark, or a sub-allocating pool
- GPU-only lists can't be spilled, activations kept across streamed blocks take every byte from the weights
- a reference in a local stays alive after its last use in a callee-saved register or spill slot (conservative stack scan)
- NVFP4 matrices (`TensorFile_Fours`) load as 8 bit rows (27 instead of 15.7 GB): e2m1 rows with their block scales on the GPU

## Loading
- 8 bit rows gathered straight from the mapped file (the 21 GB H3 DiT loads in 10-12 s, ComfyUI 4-7)
- no native test reaches the locked path of `strict_packed_rows` and `strict_rows` (matrices of 4 MB and more): a test with a big matrix

## Generated loops
- a loop reading a member of an object parameter, a type constant or `.Length` of a list parameter stays on the CPU: pass loop invariant members as numbers
- `.Round` keeps a loop on the CPU: a device Round
- bounds, reduction and tail checks remain: extend the four wide loads (`MlirBatched`) and fuse memory passes
- an index divided by a power of two (`size / 2`) divides in 64 bits at every read: shift
- convolutions of 64 bit pixels from a CPU loop miss strict_gpu_convolution without STRICT_GPU=all (H3 VAE encode of a PNG frame 12.5 instead of 0.7 s)
- each convolution writes bfloat16 copies of its input and weights first; stages of 64 channels or 3 buffers once shared memory allows it
- PNG encoding takes 0.28 s for a 720x1280 image (ComfyUI 0.08)
- small number types as kernel values: values, parameters and members of types with 2-4 Number members (Complex) and lists of them read by index work (StructLanes, LaneCalls, MlirDeviceRecords), still on the CPU: a `for` over a List(Complex), a List(Complex) result, a method returning a Complex that has a loop or an early return, the 32 bit types (Vector2, Vector3, Size); a flattened list goes up again for every launch; a `to Text` reached only through text conversion is not emitted unless something calls it (why lane inlining skips `to`)
