# GPU improvements
Details for the GPU line of [improvements.md](improvements.md). Diffusion models (Moss, H3) and GPU speed are worked on from `../StrictComfy/` (`Plan.md`, `Handoff.md`, `Performance.md`). Remove what is done, add new findings at the end of their section, bugs go into improvements.md.

## Attention
ComfyUI's comfy_kitchen `int8_attention` (`qk_int_sv_i8_attn_kernel`: Q and K rotated by a Hadamard of 128, a K scale per lane and 128 keys, int8 V scaled per number, uint8 shares, 8 warps of 16 queries and 128 keys per step at 255 registers and 48 KB) takes 47.8 ms per call of H3's 10 s clip with its tensor pipe 68.7 % busy, a step 5.24-5.29 s, Sol-Attn routing 3.34-3.49 s; Strict's `strict_device_dense_attention` (the same block shape, values transposed and scaled per number, unsigned 8 bit shares, the row sums on the tensor cores) 58.7 ms at 57 % (steps 6.00-6.04 s), `strict_device_bytes_attention` with `Attention.routing` 3.62-3.71 s (StrictComfy Performance.md). Left, from ncu (240 against 211 instructions per warp-tile of 32 positions):
- the softmax between the products of a step is 296 instructions against comfy_kitchen's 154 and both warps of a sub-partition reach it together after the barrier, so the tensor pipe idles while their 64 exponentials per lane wait for the quarter rate MUFU (`ex2.approx.f16x2` is two MUFU on Ada, measured, no gain): part of them as a polynomial on the fma pipe like FlashAttention 4, or warps 4 to 7 half a step behind warps 0 to 3
- the row maximum on the integer sums like comfy_kitchen (IMNMX on the otherwise idle integer pipe) needs one key scale per lane and step instead of one per position (comfy_kitchen: 4 per 128 keys)
- the copies of the 4 tiles of a step sit in conditional blocks, 2.7 % of the samples are long scoreboard stalls right after them: branch-free copies
- the planner reads all pooled keys of a head in every block (moments and centroid scores, about 1.5 of 15 ms routed): moments once per head, the scores as a small product kernel
- the Hadamard on Q and K
- routing plans at most 1024 key blocks (65536 positions, about 20 s of 480x832 H3), longer prompts attend densely (2.4 times slower per step): plans in global memory lift the limit

## 16 bit lists
A GPU loop of a method returning `BFloat16s` stores bfloat16, never by size alone (bfloat16 everywhere made long Qwen3.8 prompts decode NaN). Left:
- bf16 QK attention beyond the 128 wide QKV path
- attention, product and hadamard results are 32 bit
- joining lists of different formats on the GPU goes through the host (64 bit doubles of a loop reading no list and 32 bit floats, a host list after 64 bit doubles there): convert on the device
- LLVM 23 PTX of strict_device_whole(s) is ~3 % slower than LLVM 22's

## Memory budget
H3 DiT steps (21.7 GB of weights, 37 of 50 blocks resident, the rest streamed) take 3.6-3.8 s against ComfyUI's 3.3 s, ~230 ms of it idle; ComfyUI streams too but overlaps it (60 ms idle). Left:
- Moss on the 3080 (budget ~7.6 GB) re-uploads 7-11 GB of its 5.6 GB weights per image (`--quality fastest`, nsys: sampling window 1.45 s, GPU busy 1.04 s): at the peak 5.42 GB weights and ~2.2 GB of pool blocks (128/64/32/16 MiB classes, 3 x 64 MiB from the text encoder still alive, 19 x 16 MiB from the start of sampling), and the VAE decode between images peaks at ~3 GB (2 x 512, 3 x 256, 6 x 128 MiB) and evicts weights again. With 1 GB more budget image 1 uploads 2.5 GB, images 2-3 still 5-8 GB. Levers: bf16 GEMM results (`strict_gpu_tokens` has no result format, so `BFloat16s` methods still get 32 bit sums), a class between the powers of two (bf16 qkv sums of 68.2 MB take a 128 MiB block; four classes per doubling above 16 MB with lending one doubling up left the images identical but took 13.86-14.31 instead of 13.70-13.79 s for 4 images, more driver allocations), the decode's footprint (its maps are bf16 already, the biggest are 472 MB maps of 256 channels at 720x1280 plus the bf16 normalized copy `strict_gpu_convolution_normed` writes before a convolution: norm and swish on the convolution's stage load would drop that copy), and `strict_gpu_evicted`'s 2 GB headroom (512 MB: 4 images 13.6-13.9 instead of 14.1-14.3 s, H3 unmeasured).
- three uploads ahead in a ring (StrictComfy scratchpad ahead_ring.py): H3 idle 128 -> 110 ms per step, but Bonsai 27B at 10 GB decode 12.2 -> 15-16 ms from the refill walk per rows use; a ring needs a cheaper refill (check for a free upload before walking the chain)
- power of two size classes leave ~1.9 GB of H3's pool unused; finer classes churned activations through the driver, they need weights evicted first or blocks lent across classes
- without spilled rows the trim before an own block runs at every block (9.5-12 s per step with `%ownReserve` at 1 GB instead of 256 MB)
- the budget is the driver's free memory at the first allocation less 1.5 GB: on Windows that figure moved by 0.9 GB between starts next to ComfyUI (same programs, same memory), with 0.5 GB less margin H3 paged for minutes; DXGI's QueryVideoMemoryInfo budget and its change notification would give the real limit
- MossImage with ~0.9 GB used by other programs still has 1-8 full collections per step: a sub-allocating pool would end them
- GPU-only lists can't be spilled, so H3 activations kept across the 50 streamed blocks (Spectrum feature anchors) take every byte from the weights: spillable activations or an explicit move to the host
- H3's ~98 full collections per step come from strict_gpu_obtain before each 128 and 512 MB activation block (18-28 ms each in strict_gc_mark_range, 40 % of the main thread); collecting less was slower (blocks from the driver evict streamed weights), so make the mark cheaper (skip lists without references)
- a reference in a local (`let moss = Loaded(model)`) still stays in a callee-saved register or spill slot of a live frame after its last use, the conservative stack scan keeps it alive; call results passed straight on to another call are handed over through a cleared volatile slot (`MlirHanded`)
- NVFP4 matrices (H3's Qwen3-VL 32B text encoder, `TensorFile_Fours`) load as 8 bit rows: 27 GB instead of 15.7 GB, so they don't fit the 4090 whole and stream; 4 bit e2m1 rows with their 8 bit float block scales on the GPU would fit

## Loading
The 21 GB H3 DiT loads in 10-12 s (ComfyUI 4-7 s), mostly page faults of the copy into fresh memory, and its first step uploads all of it (10.1-10.6 instead of 3.7 s), now 4.3 s on a 3080 since file matrices of 64 MB (since then 4 MB) and more are locked in place as their own spill. Next: 8 bit rows gathered straight from the mapped file.
LoRAs merge on the GPU in the device copy of their rows right after its upload (`strict_gpu_tuned_cached`, the CPU `TensorFile_Tuned` is the fallback), with factors that are on the GPU already (the rotated downs, the bfloat16 ups of StrictComfy `MossLora.Packed`) read where they are: Moss with its rank 128 DMD turbo LoRA loads in 3.4 s instead of 1.2 s without it. Left: reading and packing the LoRA tensors of every linear (StrictComfy `MossLora.Loaded`, about 0.75 s), the write-back of the merged rows into their locked block (skipping it leaves the host copy stale for CPU reads and evictions), the merge kernel at about 40 % of its instruction limit.
No native test reaches the locked path of `strict_packed_rows` and `strict_rows` (matrices of 4 MB and more, only MossImage and H3 against ComfyUI): a test with a big matrix.

The first run after a runtime change waits 28.5 s for the driver to compile the 25 MB runtime PTX (cached by the driver after that). ptxas takes as long with `--split-compile 0`: a few GEMM kernels dominate (strict_device_whole_run 80K PTX lines, eights_scaled 63K, wholes 60K, whole 52K of 772K), so a cubin built with the program only moves the wait into the build. Releases should ship cubins for sm_75, sm_80, sm_86 and sm_89 next to the executable and load the one for the GPU before the PTX; splitting those kernels would shorten the compile but risks their speed.

## Generated loops
Instruction bound on the RTX 3080 (120-330 instructions per number). Left:
- a loop reading a member of an object parameter (`layout.width` in StrictComfy `H3Final.Unpatched`), a type constant or `.Length` of a list parameter (`H3Vae.Finished`) stays on the CPU: pass loop invariant members as numbers
- `.Round` keeps a loop on the CPU (StrictComfy `H3Frames.Bytes` over 148M frame numbers, `(value * 255 + 0.5).Floor` runs on the GPU): a device Round
- bounds, reduction and tail checks remain: extend the four wide loads (`MlirBatched`) and fuse memory passes
- an index with a denominator other than 1 (`size / 2` inlined in MossBlock.Rotated) divides in 64 bits at every read (82 instead of 66 us): shift power of two denominators
- without STRICT_GPU=all StrictComfy's H3 VAE encode of a PNG frame (64 bit pixels of a CPU loop) takes 12.5 instead of 0.7 s: its convolutions miss strict_gpu_convolution

## MOSS
Warm 720x1280 VAE decode on a 3080 0.27-0.28 s like ComfyUI's 0.27 s since `strict_device_convolutions` copies bfloat16 stages asynchronously (50-58 instead of 19-24 TFLOPS). Left: the bfloat16 input and weights each convolution writes first (a list as big as its input, freed by the collector), stages of 64 channels or 3 buffers once the shared memory allows it, PNG encoding 0.28 against 0.08 s, cold loading (`../StrictComfy/Performance.md`).

Moss's Qwen3.5 9B text encoder keeps its tokens (`Safetensors.Quantized(name, 32)`), the 8 bit tensor cores code them twice (about 15 bits) from 16 tokens on. Prompts below 16 tokens still multiply 32 bit tokens (0.3 s for 12 tokens).
