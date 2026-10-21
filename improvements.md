# Improvements
In priority order, short and actionable. Remove what is done (results go into the README of the package), add what is found.

## Next
1. **CI complete and fast**: `.github/workflows/ci.yml` runs the bootstrap, one self build and the VM tests of the fast folders (about 5 minutes, 10 at most). Add `Examples`, `Validators`, `Runtime`, `Expressions` and native builds back while it stays under 10 minutes; a failing native build prints the file and exit code. Faster compiler work: a self build takes about 2 minutes, `strict test .` and `strict test Compiler` over 4 minutes each without `.strictbinary` caches.
2. **Diffusion models in Strict**: Moss on the RTX 3080, then MiniMax H3 on the RTX 4090, later a ComfyUI node. Plan in `../StrictComfy/Plan.md`.
3. **Qwen3.8 for the robot arm**: the prefill (1.83 ms per token at 460 tokens) with shared memory tiles for `Products`; its MTP layer as the drafter of `Speculation` (lookup drafts get 1.02 tokens per step, Ollama 0.40.4 drafts with it and decodes at 11.3 instead of 21.5 ms per token); batches (`Copies`); speech to text, a small Strict endpoint the arm calls, the time from question to first sound. A loop calling `Length(start)` in `Corrected` stays on the CPU (find the rule in `MlirGpuCheck`).
4. **GPU speed on the RTX 4090** (Qwen3-8B Q4_K_M: decode 6.8 ms, prefill 0.15 ms per token, llama.cpp 0.10): fewer launches per layer (rotations with the cache append, 8 bit inputs written by the producing kernels instead of 145 quantize kernels); 128x128 tiles with 8 warps, Q6_K and 8 bit rows in the tiles; continuous batching, paged and prefix caches. Check why Qwen3-8B prefill is 0.24-0.30 ms since 2026-10-21 evening.
5. **Big models and memory**: Qwen3.8 is live at 17.4-19 GB after a full collection (Ollama 18.2 GB used at 4K context) but fills the card to 23.5 GB: 4-6.7 GB of free 2 MB+ blocks stay in the pool (power of two classes), DeltaNet states (3.1 MB in 4 MB blocks, 48 per token) are promoted by young collections and only freed by full ones (90-110 ms each, scanning about a million small objects, mostly the tokenizer). A 256 or 128 MB reserve starves 1.4-2.5x more often (slower), keep 512. Qwen3-Coder-30B-A3B with all 128 experts: free a layer's temporaries before the next one, drop the `Joined` and `Stacked` per expert row copies (`List.Slice` copies), return 64 MB chunks, index the residents table instead of a linear search. Grouped expert products read and write through the order instead of copying rows; the first 18.6 GB upload is bound by the CPU copy.
6. **Language model research** (results in `NeuralNetworks/README.md`): scoring on the GPU (logits download 4-5 ms per token), Strict coding tasks timed until their tests pass, weight bytes, peak VRAM and time to first token next to every perplexity; training all weights (LoRA or full, GPT-2 steps take 2.7 s) and training after pruning; Qwen3 layer, neuron and head pruning with measured speed (narrowed rows stay packed); experts ranked and removed in Strict; speculation without cache copies for rejected drafts and with exact acceptance when sampling. Later: parallel method generation against a shared contract, smaller vocabularies, next-concept modeling, HybridAgent latent reasoning.
7. **Image blur** (`ImageProcessing/BenchBlur.strict`): a whole parallel child for collecting Colors on the CPU (C reaches 108 ms on one thread with the same hoisted check), lane packed types in generic GPU kernels, member-wise `+ - * /` from Automatic for every type made only of numbers, the CPU share overlapped with the GPU download.
8. **Web and other platforms**: run on real phones (1 GB initial wasm memory is likely too much for iOS), smaller downloads (SoulCraft 150 MB), content hashes in file names, sound without a first click, saves on IDBFS, RocketCommander and Breakout below 60 fps; run on Apple Silicon, notarization, Metal shared memory and a real allocator; Android, iOS, Steam.
9. **AMD GPUs**: about 250 us of host work per launch (GPT-2 48 ms per token on the RX 480), `sdot4`, WMMA and MFMA, the 512 thread tiles need more than the 64 KB of an AMD workgroup, ROCm on Linux.

## Bugs
- `List.Average` of Colors clamps natively (`Color.test.exe`, Alpha 127): needs wide sums per member.
- A `let` from the inner index in a warp summed loop of a 32 bit kernel reads wrong positions (RTX 4090, chunked `DeltaBlock`).
- A `Generic` parameter of a non-generic type is not specialized (`Unsupported native call Strict/Generic.Area`), `strict check` misses it.
- Unpacked rows of a matrix stay pinned while the model lives.
- `strict test NeuralNetworks lines` crashes natively on macOS (`strict_list_restoring` reaches a null link).
- `strict test Examples` crashed once at start (segfault, 2026-10-21), not reproduced.
- `strict test` prints `Unsupported native call` for a test calling a method with too many arguments and still reports all tests passed (DeltaBlock, 2026-10-21).

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Qwen3-Coder-Next 80B-A3B pruned to 18-20 GB: needs more VRAM and the `qwen3next` architecture.
