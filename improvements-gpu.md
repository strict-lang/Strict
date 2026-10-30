# GPU improvements
Details for the GPU line of [improvements.md](improvements.md), one short line each, measurements go into commit messages. Diffusion models (Moss, H3) live in `../StrictComfy/`. Remove what is done, bugs go into improvements.md.

## Attention
- `strict_device_dense_attention` is ~20 % slower than comfy_kitchen's `int8_attention` (softmax 296 against 154 instructions per step, the tensor pipe idles on the quarter rate MUFU): part of the exponentials as a polynomial on the fma pipe, or warps 4 to 7 half a step behind
- the row maximum on the integer sums (IMNMX) needs one key scale per lane and step instead of one per position
- the 4 tile copies of a step sit in conditional blocks (long scoreboard stalls after them): branch-free copies
- the routing planner reads all pooled keys of a head in every block: moments once per head, the scores as a small product kernel
- the Hadamard on Q and K
- routing plans at most 1024 key blocks (65536 positions), longer prompts attend densely: plans in global memory
- bfloat16 Q·K runs at half the GeForce tensor rate; 8 bit Q·K moves Moss images 22-24 dB: smooth K like SageAttention
- head size 64 (H3 VAE decode) has no tensor core attention kernel

## 16 bit lists
- bf16 QK attention beyond the 128 wide QKV path
- attention, product and Hadamard results are 32 bit
- joining lists of different formats on the GPU goes through the host: convert on the device
- LLVM 23 PTX of strict_device_whole(s) is ~3 % slower than LLVM 22's
- compiled loops index in 64 bits: 32 bit index math where the counts fit

## Memory
- streamed weights are not overlapped with compute (H3 idles ~230 ms per step, ComfyUI 60): uploads ahead in a ring need a cheaper refill than walking the chain
- power of two size classes leave ~1.9 GB of H3's pool unused, finer classes churned activations through the driver
- without spilled rows the trim before an own block runs at every block
- full collections before big activation blocks (H3 ~98 per step, 18-28 ms each in strict_gc_mark_range): skip lists without references in the mark, or a sub-allocating pool
- GPU-only lists can't be spilled, activations kept across streamed blocks take every byte from the weights
- a reference in a local stays alive after its last use in a callee-saved register or spill slot (conservative stack scan)
- NVFP4 matrices (`TensorFile_Fours`) load as 8 bit rows (27 instead of 15.7 GB): e2m1 rows with their block scales on the GPU

## Loading
- 8 bit rows gathered straight from the mapped file (the 21 GB H3 DiT loads in 10-12 s, ComfyUI 4-7)
- the bfloat16 and 32 bit ups paths of `strict_device_tuned` give slightly different merged rows: the same arithmetic in both
- no native test reaches the locked path of `strict_packed_rows` and `strict_rows` (matrices of 4 MB and more): a test with a big matrix
- releases should ship cubins for sm_75, sm_80, sm_86 and sm_89 and load the one for the GPU before the PTX (the driver compiles MossImage's PTX 4 + 12-14 s on a first run)

## Generated loops
- a loop reading a member of an object parameter, a type constant or `.Length` of a list parameter stays on the CPU: pass loop invariant members as numbers
- `.Round` keeps a loop on the CPU: a device Round
- bounds, reduction and tail checks remain: extend the four wide loads (`MlirBatched`) and fuse memory passes
- an index divided by a power of two (`size / 2`) divides in 64 bits at every read: shift
- convolutions of 64 bit pixels from a CPU loop miss strict_gpu_convolution without STRICT_GPU=all (H3 VAE encode of a PNG frame 12.5 instead of 0.7 s)
- each convolution writes bfloat16 copies of its input and weights first; stages of 64 channels or 3 buffers once shared memory allows it
- PNG encoding takes 0.28 s for a 720x1280 image (ComfyUI 0.08)
- prompts below 16 tokens multiply 32 bit tokens (Moss's text encoder, 0.3 s for 12 tokens)
