# GPU improvements
Details for the GPU line of [improvements.md](improvements.md), one short line each, measurements go into commit messages. Diffusion models (Moss, H3) live in `../StrictComfy/`. Remove what is done, bugs go into improvements.md.

## Attention
- `strict_device_dense_attention` keeps the 8 bit tensor pipe 53 % busy (Moss on the 3080, 1.26 ms per layer, 253 registers leave one block per SM; comfy_kitchen's `int8_attention` was ~20 % faster on the 4090 before warps 4 to 7 ran half a step behind): fewer softmax instructions (part of the exponentials as a polynomial on the fma pipe), or two blocks per SM
- the row maximum on the integer sums (IMNMX) needs one key scale per lane and step instead of one per position
- the 2 tile copies of a routed step sit in conditional blocks (long scoreboard stalls after them): branch-free copies like `strict_device_dense_issued`
- the routing planner reads all pooled keys of a head in every block: moments once per head, the scores as a small product kernel (2026-11-08: fusing the moments and score passes, or only storing the centroid before the moments loop, made the routed kernel run 2.25x the instructions and L2 bytes at 480x832, so plans got longer although the data flow reads as equivalent: dump plan lengths before trying again)
- the Hadamard on Q and K
- routing plans at most 1024 key blocks (65536 positions), longer prompts attend densely: plans in global memory
- head size 64 (H3 VAE decode) has no tensor core attention kernel
- routed strict_device_bytes_attentions is L2 bound on the 4090 (L2 81 %, DRAM 4.5 %, compute 40 %, 238 registers): every 64 query block loads its own tiles, kernel time is ~0.7 ms plus ~3.2 ms per plan at 384x640 (half the entries: 3.88 -> 2.29 ms); neighbouring query blocks of a head share 46-118 of their 78-134 exact tiles, so two query blocks per block of 8 warps loading the union once (each half computing only its own plan, pooled chunks are per half) would load 27-46 % less: ~25-30 % kernel, ~10-12 % of an H3 step at 480x832, ~4 % at 384x640

## 16 bit lists
- bf16 QK attention beyond the 128 wide QKV path
- attention, product and Hadamard results are 32 bit
- joining lists of different formats on the GPU goes through the host: convert on the device
- LLVM 23 PTX of strict_device_whole(s) is ~3 % slower than LLVM 22's
- compiled loops index in 64 bits: 32 bit index math where the counts fit

## Memory
- streamed weights are not overlapped with compute (H3 idles ~230 ms per step, ComfyUI 60): uploads ahead in a ring need a cheaper refill than walking the chain
- without spilled rows the trim before an own block runs at every block
- GPU-only lists can't be spilled, activations kept across streamed blocks take every byte from the weights
- a reference in a local stays alive after its last use in a callee-saved register or spill slot (conservative stack scan)
- NVFP4 matrices (`TensorFile_Fours`) load as 8 bit rows (27 instead of 15.7 GB): e2m1 rows with their block scales on the GPU

## Loading
- 8 bit rows gathered straight from the mapped file (the 21 GB H3 DiT loads in 10-12 s, ComfyUI 4-7)
- no native test reaches the locked path of `strict_packed_rows` and `strict_rows` (matrices of 4 MB and more): a test with a big matrix

## WebGPU
- strict_convolution runs at ~6 TFLOPS on a 3080 (30 peak): not limited by workgroup bank conflicts, index divisions, register prefetch or the rolled inner loop (unrolled it was 6 % faster in a decode, products slower, shaders compiled longer)

## Generated loops
- a `for` over candidates times examples whose body calls a method with an inner counted `for` and a mutable accumulator (a small opcode interpreter, Ben's candidate evaluation) gets no kernel and stays a CPU parallel loop (800000 evaluations in 3 ms there): kernels for such bodies, then batches big enough for the GPU
- a loop reading a member of an object parameter, a type constant or `.Length` of a list parameter stays on the CPU without a warning (one in a kernel body kept Convolved_Summed off the GPU, a taef2 decode took 2300 s of CPU): pass loop invariant members as numbers, or warn
- `.Round` keeps a loop on the CPU: a device Round
- bounds, reduction and tail checks remain: extend the four wide loads (`MlirBatched`) and fuse memory passes
- an index divided by a power of two (`size / 2`) divides in 64 bits at every read: shift
- convolutions of 64 bit pixels from a CPU loop miss strict_gpu_convolution without STRICT_GPU=all (H3 VAE encode of a PNG frame 12.5 instead of 0.7 s)
- each convolution writes bfloat16 copies of its input and weights first; stages of 64 channels or 3 buffers once shared memory allows it
- PNG encoding takes 0.28 s for a 720x1280 image (ComfyUI 0.08)
- small number types as kernel values: values, parameters and members of types with 2-4 Number members (Complex) and lists of them read by index work (StructLanes, LaneCalls, MlirDeviceRecords), still on the CPU: a `for` over a List(Complex), a List(Complex) result, a method returning a Complex that has a loop or an early return, the 32 bit types (Vector2, Vector3, Size); a flattened list goes up again for every launch; a `to Text` reached only through text conversion is not emitted unless something calls it (why lane inlining skips `to`)
- Constant lists read in a method a GPU loop calls (FluxPreview.Channel's Factors and Bias) keep the loop on the CPU natively: pass them to the kernel like the members it reads.
- Web collections: every 1 GB of WebGPU results runs a full collection (about 18 per Moss step at 480p, 0.6 s), a young one alone does not give the buffers of dead results back (video memory piled up); reclaiming uncounted lists first frees them for 0.03 s, but steps were not faster: the collections overlap the queued GPU work.
- Uniform branches still in GPU loops, measure their share first (none showed up in Qwen3-Coder): `tokens_blocks` switches on the row bits per block, `products_run` runs bits chains per element, `strict_device_dots` has no bfloat16 arm, prompt attention masks every tile.
