# GPU improvements
Details for the GPU line of [improvements.md](improvements.md). Diffusion models (Moss, H3) and GPU speed are worked on from `../StrictComfy/` (`Plan.md`, `Handoff.md`, `Performance.md`). Remove what is done, add new findings at the end of their section, bugs go into improvements.md.

## Attention
ComfyUI's comfy_kitchen `int8_attention` (Q and K rotated by a Hadamard of 128, int8 V, uint8 shares) runs H3's 10 s step in 5.24-5.29 s, Sol-Attn routing in 3.34-3.49 s; Strict's `strict_device_bytes_attention` (8 bit keys, values and shares, 64 tokens per block) 6.89-7.04 s dense and 3.62-3.71 s with `Attention.routing` (StrictComfy Performance.md). Left, from ncu (dense 85 ms per call, tensor pipe 48 % busy, 2 blocks of 4 warps per SM at 238 registers and 43 KB):
- 3 blocks per SM like Sol-Attn need at most 168 registers (spills at 168 made it slower) and 33 KB: rows of 128 bytes with a swizzle instead of 144, plans in global memory from a planning kernel
- values transposed and scaled per number by their quantize pass: no byte permutes after ldmatrix.trans and no scale per position on the shares (about 75 of 446 instructions per tile of 32 positions and warp)
- the planner reads all pooled keys of a head in every block (moments and centroid scores, about 1.5 of 15 ms routed): moments once per head, the scores as a small product kernel
- the Hadamard on Q and K
- routing plans at most 1024 key blocks (65536 positions, about 20 s of 480x832 H3), longer prompts attend densely (2.4 times slower per step): plans in global memory lift the limit

## 16 bit lists
A GPU loop of a method returning `BFloat16s` stores bfloat16, never by size alone (bfloat16 everywhere made long Qwen3.8 prompts decode NaN). Left:
- bf16 QK attention beyond the 128 wide QKV path
- attention, product and hadamard results are 32 bit
- joining lists of different formats goes through the host
- LLVM 23 PTX of strict_device_whole(s) is ~3 % slower than LLVM 22's

## Memory budget
H3 DiT steps (21.7 GB of weights, 37 of 50 blocks resident, the rest streamed) take 3.6-3.8 s against ComfyUI's 3.3 s, ~230 ms of it idle; ComfyUI streams too but overlaps it (60 ms idle). Left:
- three uploads ahead in a ring (StrictComfy scratchpad ahead_ring.py): H3 idle 128 -> 110 ms per step, but Bonsai 27B at 10 GB decode 12.2 -> 15-16 ms from the refill walk per rows use; a ring needs a cheaper refill (check for a free upload before walking the chain)
- power of two size classes leave ~1.9 GB of H3's pool unused; finer classes churned activations through the driver, they need weights evicted first or blocks lent across classes
- without spilled rows the trim before an own block runs at every block (9.5-12 s per step with `%ownReserve` at 1 GB instead of 256 MB)
- MossImage with ~0.9 GB used by other programs still has 1-8 full collections per step: a sub-allocating pool would end them
- GPU-only lists can't be spilled, so H3 activations kept across the 50 streamed blocks (Spectrum feature anchors) take every byte from the weights: spillable activations or an explicit move to the host
- H3's ~98 full collections per step come from strict_gpu_obtain before each 128 and 512 MB activation block (18-28 ms each in strict_gc_mark_range, 40 % of the main thread); collecting less was slower (blocks from the driver evict streamed weights), so make the mark cheaper (skip lists without references)
- a reference in a local (`let moss = Loaded(model)`) still stays in a callee-saved register or spill slot of a live frame after its last use, the conservative stack scan keeps it alive; call results passed straight on to another call are handed over through a cleared volatile slot (`MlirHanded`)

## Loading
The 21 GB H3 DiT loads in 10-12 s (ComfyUI 4-7 s), mostly page faults of the copy into fresh memory, and its first step uploads all of it (10.1-10.6 instead of 3.7 s), now 4.3 s on a 3080 since file matrices of 64 MB and more are locked in place as their own spill. Next: 8 bit rows gathered straight from the mapped file.
Matrices below 64 MB (Moss) still go up through the staging buffer and no native test reaches the 64 MB path of `strict_packed_rows` (only H3 against ComfyUI): a test with a big matrix, then a lower threshold if locking pays off there.

The first run of a new build waits for the driver to compile the 20 MB runtime PTX on one thread (MossImage: 114 s); `ptxas -arch=sm_89 --split-compile 0` takes 43 s: write a cubin for the local GPU at build time and load it before the PTX, and split the largest functions (strict_device_whole_run, wholes, whole, eights_scaled: 41-63k PTX lines each).

## Generated loops
Instruction bound on the RTX 3080 (120-330 instructions per number). Left:
- a loop reading a member of an object parameter (`layout.width` in StrictComfy `H3Final.Unpatched`) stays on the CPU: pass loop invariant members as numbers
- bounds, reduction and tail checks remain: extend the four wide loads (`MlirBatched`) and fuse memory passes
- an index with a denominator other than 1 (`size / 2` inlined in MossBlock.Rotated) divides in 64 bits at every read (82 instead of 66 us): shift power of two denominators

## MOSS
Warm generation 4.10-4.18 s against ComfyUI's 3.51 s with sampling at parity: VAE decode 0.63 against 0.27 s, PNG encoding 0.28 against 0.08 s, cold loading (`../StrictComfy/Performance.md`).
