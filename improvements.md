# Improvements
In priority order. Remove what is done, add what is found, keep each item to what is still open (results go into the README of the package). Every item gets done before the bigger image processing and neural network examples, parallel loops and GPU (decided 2026-10-12).

1. **Specialized language models: pruning and parallel generation**

    Goal: extreme measured speed and small resident models for English text and Strict code.
    Accept some accuracy loss and loss of unrelated languages, tasks and general knowledge,
    provided held-out English and Strict generation remain useful. Prefer parallel execution;
    optimize time to useful output and passing code, not only raw tokens/s.
    Results belong in `NeuralNetworks/README.md`; keep this entry to remaining experiments.

    - **Order.** Always a small trained model first (distilgpt2 in `distilgpt2/`), then GPT-2,
      then Qwen3-0.6B, only then 8B and 30B (`NeuralNetworks/qwen3/` has Qwen3-0.6B Q8_0 and
      Q4_K_M and Qwen3-8B Q4_K_M, Ollama's `qwen3-coder:30b` blob is the Qwen3-Coder-30B-A3B GGUF).
      Done and measured (`NeuralNetworks/README.md`): held-out perplexity and bits per character
      (`Evaluation`, `Perplexity`), greedy layer dropping, MLP neuron pruning with bias
      compensation, lossless prompt lookup speculation with verify blocks up to 16 tokens, BitFit
      training of GPT-2. Methods Strict does not have yet are tried in a PyTorch prototype first
      (it reproduces the Strict perplexities), only the winners get built in Strict.

    - **Measurement.** Open: scoring on the GPU (every position downloads its logits, 4-5 ms per
      token), unseen Strict coding tasks with executable assertions and the time until they pass,
      weight bytes, peak VRAM, cold load and time to first token next to every perplexity.
      RTX 3080 baseline (10 GB, 2026-10-19, `Qwen3Benchmark` defaults, 10 prompt tokens, 100 steps):
      Qwen3-0.6B Q4_K_M median 2.58 ms (387 tokens/s), batch 4 3.46 ms (1155 tokens/s), Q8_0 2.84 ms; the
      logits of one token row by row (2026-10-20): Q4_K_M 2.30 ms (435 tokens/s), Q8_0 2.63 ms.

    - **Training.** Done: BitFit for GPT-2 (`Gpt2Training`, `Finetuning`, README: 100 steps make GPT-2
      2.2 times better at Strict and English 27% better, distilgpt2 2.3 times better at Strict). Open:
      a step takes 2.7 s for GPT-2 (distilgpt2: forward 20 ms, logits and their errors 120-180 ms, the
      backward of a block about 200 ms), training the weights themselves (gradients of the
      products or LoRA, PyTorch: all weights 34.9 and 10.6 against BitFit 33.4 and 11.1), training
      pruned models (`Gpt2.Without` or `NeuronPruning`, then `Finetuning`; PyTorch without 4 blocks:
      60.7 and 20.5), distilling from the trained model, Qwen3 (RMS norms, rotary positions, grouped
      queries, gated MLPs, 4 bit rows), more data and steps. References:
      [Minitron](https://arxiv.org/abs/2407.14679), its [report](https://arxiv.org/html/2408.11796v4),
      [BitFit](https://arxiv.org/abs/2106.10199).

    - **Whole-layer pruning.** Done greedily by calibration perplexity (GPT-2 tolerates one block,
      distilgpt2 none). Open: ShortGPT's hidden state similarity as the cheaper ranking to compare
      with, measured decode time without the blocks, Qwen3-0.6B and 8B. Reference:
      [ShortGPT](https://arxiv.org/abs/2403.03853).

    - **MLP neuron pruning.** Done like [FLAP](https://arxiv.org/abs/2312.11983) (results in the
      README). Prototype: with block 6 dropped first GPT-2 without 10% and 20% of the neurons gets
      45.3 and 29.8, 51.4 and 31.0 (72.8M and 67.6M block weights, dropping blocks 6 and 1 instead:
      47.0 and 30.7 at 70.9M). Open: keep the narrowed contraction rows packed (they are doubles
      now), the gated MLPs of Qwen3, measured speed, attention heads the same way.

    - **Width pruning with SliceGPT.** Prototype only, not worth building for GPT-2: LayerNorm stays
      when the rotation keeps the all-ones direction as one basis vector with the same weight in
      every coordinate (centering then commutes with it), the tied head needs the hidden state
      rotated back to 768 (the sliced embedding as head gives perplexity 14 million). One rotation
      per norm input from the already sliced model gets GPT-2 10%: 49.2 and 27.1, 20%: 58.4 and 31.1,
      30%: 76.5 and 37.5, but the k by k rotations of the residual make 10% larger than the original
      (88.0M block weights instead of 85.1M) and 30% only 22% smaller; one rotation for all layers
      needs none but loses more (20%: 66.8 and 35.5 at 68.0M), one per block breaks (2306). Dropping
      blocks or neurons wins at the same size. Qwen3 (RMS norms need no trick, an untied head is
      sliced too; English and Strict perplexity): 0.6B 35.6 and 5.40, 10% 52.7 and 7.36, 20% 84.3 and
      11.05 (6K calibration tokens). 8B 9.36 and 3.47 (8.19B weights): with 50K calibration tokens
      10% 13.03 and 3.71 (8.35B, the rotations outweigh the slices), 20% 18.61 and 4.19 (7.33B), with
      6K tokens 10% 20.4 and 4.10, 20% 39.1 and 6.09, so calibration size matters most (the paper
      uses 2M). Qwen3-Coder-30B-A3B 2.06 and 3.18 (30.5B, it knows Sherlock Holmes by heart): 10%
      5.08 and 3.27 (27.8B), 20% 12.95 and 3.63 (24.7B), Strict code survives, memorized English
      does not. Open: more calibration (only Strict code for a Strict model), BitFit or full
      training after slicing, the rotations only where width matters. Reference:
      [SliceGPT](https://arxiv.org/abs/2401.15024).

    - **Mixtures of experts (Qwen3-Coder-30B-A3B, Qwen3-Coder-Next 80B-A3B).** Done (README): batched
      experts, prompts grouped by expert with one launch per matrix (`Grouping`, `Batch.Grouped`),
      `MixturePruning` (experts per token, layers), expert pruning by rewriting the GGUF (prototype
      `gguf_prune.py`: 96 of 128 experts cost Strict +1%, 14.2 GB). Open: the prefill is 0.17 ms per
      token against 0.12 for llama.cpp, but a full collection lands in nearly every prefill (0.4 ms
      then, see Language and compiler); the padded rows are bounded by pairs + 63 per expert (about 30%
      more expert rows at 800 tokens than the exact count, which needs a download); `Grouping.Gathered`
      and `Unsorted` copy every token row (8 and 7 ms of a prefill) where the grouped products could
      read and write through the order; below 64 tokens (speculative verification) the flat experts
      stay (10 tokens 1.59 ms per token, grouped 6.2). CUDA graphs: launching directly after a graph
      that was not reused broke decoding (its segments drift against the steps and were never found
      again). Ranking and removing experts in Strict instead of the prototype.

    - **Attention heads, low rank and sparsity.** Test head removals by perplexity impact;
      physically shrink Q/K/V and output-projection slices, respecting grouped-query sharing.
      Compare with MLP pruning at equal quality loss. Try activation-aware low-rank factors
      for attention and MLP matrices separately; do not assume either compresses better.
      A matrix of shape m by n replaced by ranks r saves weights only if r(m+n) < mn;
      measure the extra launch/intermediate costs. Reference:
      [SVD-LLM](https://arxiv.org/abs/2403.07378). Defer 2:4 sparsity until dense pruning wins
      are measured: zeroing weights does not accelerate existing dense kernels, and sparse
      tensor-core execution needs suitable compiler/emitter support and measured quality.

    - **Vocabulary specialization later.** Test corpus-selected 16K, 8K and 4K vocabularies,
      retaining required characters, whitespace, punctuation and control tokens. Remove actual
      embedding/output rows and update tokenizer merges/IDs consistently. Reject configurations
      whose fragmentation makes useful output slower. Smaller token IDs alone remove no weights.
      Keep 256-entry vocabularies as an aggressive endpoint, not the initial target. Reference:
      [Efficient Vocabulary Reduction](https://aclanthology.org/2025.coling-industry.64.pdf).

    - **Speculative decoding.** Done with drafts looked up in the context, verify blocks up to 16 tokens
      and the decode attention for them (README: Qwen3-8B copying 4.5 times faster, Qwen3-0.6B copying
      3.7 and prose 2.7 times). Open: Qwen3-Coder-30B-A3B looks up slower than it decodes (2026-10-20,
      copying 47-60 instead of 112-144 tokens/s at 1.38 tokens per step, prose 96-125 instead of 175-188:
      a verify block of k tokens reads up to k times the experts and every rejection copies the caches of
      all 48 layers; with GPU lists freed by their counts it stays at 55-68, so the verify cost decides), so mixtures need drafts only when they
      are likely accepted; the products with the weights for 2 to 16 tokens
      (`strict_device_blocks`, tensor cores) reach 470-750 GB/s against 775-885 of single token rows
      (dp4a), decoding each part of a row once for up to 8 tokens (like llama.cpp's `mul_mat_vec_q`)
      would make 16 tokens about 7.7 instead of 10.4 ms; about 3 ms per step on the CPU (graphs
      instantiated again when the block length changes, the predictions downloaded twice); cutting
      rejected drafts without copying the cache (attention over the cache and the block as two lists,
      only accepted keys and values appended); sampling with exact acceptance (a looked up draft is a
      point mass, accepted with the target's probability); a draft model only pays when it is much
      cheaper than Qwen3-0.6B for the 8B (2.5 against 7.4 ms per token, at 65% acceptance about 120
      instead of 136 tokens/s), HyperQwen drafts with a 1 layer MTP head or a 4 bit 5 layer model
      with a 40K vocabulary. References:
      [speculative decoding](https://arxiv.org/abs/2211.17192),
      [HyperQwen](https://github.com/syv-ai/qwen38-27b-rtx3090) (Qwen3.8-27B on one RTX 3090:
      MTP and DFlash2 drafters, a 40K draft vocabulary, context lookup, 127 tokens/s single user).
      Qwen3.8 is a hybrid of Gated DeltaNet and attention layers, `Qwen3` loads it but cannot cut its recurrent memory yet.

    - **Parallel methods and files.** Generate shared members, signatures, invariants and required
      behavior first; batch independent implementations against that contract using one loaded
      model. Assemble, check and test, then repair only failures with relevant diagnostics.
      Schedule ready tasks by dependencies, refill finished batch slots and pass compact interface
      updates between stages. Sequences cannot see each other's new output automatically;
      changing earlier context invalidates its cached suffix. Compare serial generation with
      batches 2/4/8/16/32 and report total time to a passing package, including integration costs.
      Investigate [APAR](https://arxiv.org/abs/2401.06761) for learned branching only after an
      externally scheduled baseline. Combine batching with speculation and compression only
      after each wins separately. Transfer successful methods to larger locally runnable models;
      distinguish reduced resident weight size from reduced active work in mixtures of experts.
    - **Next-concept modeling, later** (added 2026-10-20). [NCM](https://arxiv.org/abs/2610.14162)
      ([code](https://github.com/HazyResearch/ncm), MIT): a multiscale residual VQ-VAE turns a text of fixed
      length L into K levels of codes (coarse to fine, the last one code per token, codebooks of 4096-8192),
      a transformer predicts all codes of a level in one pass, so a whole sequence takes K passes (8-32)
      instead of L. Measured only against diffusion models (H100, 10-31 times faster at similar quality);
      autoregressive models still win on quality (TinyStories 8.89 against 15.41, OpenWebText 35.45 against
      83.12 generative perplexity), Sudoku 86% solved on hard against 81% of diffusion. Fits the parallel
      goal and Strict methods (at most 13 lines of 120 characters, so a method is a natural fixed L, a
      prompt as prefix LM), but it is trained from scratch per dataset (tokenizer frozen first, generators
      of 25-725M), cannot reuse the Qwen3 or GPT-2 weights and needs training of all weights, which Strict
      cannot do yet. Order: after the weights themselves train, a PyTorch prototype from their repository
      on Strict methods, measured as the share of generated methods that pass their tests and the time to
      them against Qwen3-0.6B with lookup speculation; only a winner gets built in Strict (codebook lookups,
      block-diagonal and prefix attention masks in the attention kernel).
    - **Latent reasoning with HybridAgent, inference only** (added 2026-10-20).
      [HybridAgent](https://arxiv.org/abs/2610.18861) ([code](https://github.com/togethercomputer/HybridAgent), MIT):
      Qwen3-8B and 14B Base checkpoints that think partly without tokens: after the sampled anchor
      `<latent>` the last hidden state (after the final norm, `hidden_states[-1]`) goes through HybridLink
      `post_ln(h + W2 gelu(W1 pre_ln(h)))` (two d by d matrices with biases, two LayerNorms, about 34M
      weights for 8B) and becomes the next input instead of a token embedding, 4 to 200 latent steps
      until the nearest token embedding by cosine is `</latent>`. Reported 35-55% fewer steps and 63-73%
      less decode compute than text reasoning at higher accuracy (AIME, GPQA, search agents), measured as
      FLOPs on H100 estimates, not tokens/s; no coding benchmark. Training needs 8-32 GPUs, so only the
      released `HybridAgent-8B-Agentic` and `14B` (Hugging Face bf16 plus `adapter.pt`) are usable here:
      they need loading Qwen3 from safetensors or converting to Q4_K GGUF with the two anchor tokens, a
      `Qwen3.Steps` from an input vector instead of a token, the adapter from existing `Layer` and norm
      code, and the exit check as one product with the embedding table per latent step (as costly as the
      logits). Latent steps cannot be drafted, so lookup speculation only helps the text parts. Worth it
      for the robot arm's complex questions (14B Q4_K is about 9 GB) once `Qwen3` decodes cleanly on the
      4090; measure time to a correct answer against plain Qwen3-8B thinking.

## Language and compiler
- **A bare `return` in a method crashes the bytecode generator** (2026-10-21): `if ...` with `return` and no value in `Run` fails with IndexOutOfRange in `ErrorCodegen.Statement` (`node.children(0)` of a Return without children), seen in `ImageProcessing/BenchBlur.strict`, which uses if/else instead.
- **Image blur: as fast as simple C from simple Strict code** (2026-10-21, `ImageProcessing/BenchBlur.strict` against `ImageProcessing/Compare/blur.c`, .rs, .cs, .ts, .py and the parallel and NumPy versions, all the same raw RGBA 9x9 box blur of the 4096x2048 landscape with the same output bytes). BenchBlur averages the Colors of the neighbors inside the image, `Neighbors(bitmap, index).Average`, no mutables and no hand optimizations. Done: integer floor division, `for` sums of a type with its own `+` (`TypeSums`), Bitmap of Colors, `Average` fused per member (`Fusion`, `FusedAverage`), Color as one whole number of byte lanes (`MlirLanes`), whole functions with nested range loops, lists of lane values, instances (`MlirWholeInstance`, `MlirWholeFields`) and i64 slots next to f64 ones. Measured (best of 3, i9-13900KF): C 285 ms on one thread and 48.4 with OpenMP, Strict 441 and 72.4 (24 threads automatically), Rust 456 and 85.4, C# 815 and 129, NumPy 1623, TypeScript 1860. Open, in order: the GPU for lane packed types (see below, BenchBlur runs on the CPU with the GPU on); the border pixels keep `IsInside`, the inner image without it (removing every check gives about 20%: 500 to 384 ms on one thread, needs a range analysis of the condition to be sound); a whole parallel child for collecting Colors (one guard per chunk, members read once) instead of the generic child calling the whole function per pixel; List(Color) as 4 bytes per element instead of doubles; member-wise `+ - * /` for every type made only of numbers written by Automatic (Color has them by hand for now), values clamped only when they reach user code; the GPU threshold of 4M numbers may be too high for smaller loops.
- **GPU kernels do not support lane packed types yet** (found 2026-10-21): a Color is one whole number of byte lanes (`MlirLanes`); a GPU loop over Colors (BenchBlur, Ppm) scrambled them ((0, 0, 255, 40) instead of (13, 27, 41, 255) with `STRICT_GPU=all`, 32 bit kernels and lists kept as 32 bit floats on the GPU lose the low lanes, the device code has no lane construction or member reads). Loops touching such values stay on the CPU (`MlirLanes.IsTouched` in `MlirGpuCheck.IsGpu`), lists of them come back as doubles (`MlirLayout.Packed` 64). Open: lanes in 64 bit kernels (or as 32 bit integers), then BenchBlur on the GPU.
- **`List.Average` of Colors clamps natively** (found 2026-10-21, `ImageProcessing/Color.strict` test `(Color(0, 0, 0), Color(2, 4, 6)).Average is Color(1, 2, 3)` fails in `Color.test.exe` with Alpha 127): a native Color is one whole number of byte lanes (`MlirLanes`), so `Sum` through `Color.+` clamps 255 + 255 at 255 before `/ Length`; the Strict VM keeps the wide sum. `Method(...).Average` of an own method is fused per member (`Fusion`) and right; `List.Average` itself needs the same: wide sums per member for a type made only of numbers, the type built after dividing.
- **Named loop variables are not implemented** (found 2026-10-21): `for y in Range(-4, 5)` (README "Custom enumerator variables") gives `NameNotFound: y` on the Strict VM; `ImageProcessing/BenchBlur.strict` uses `value` and `outer`.
- **`Byte.Floor` has no native route on the Strict VM** (found 2026-10-21): `(Red / divisor).Floor` with a Byte member logs "Unsupported native call Strict/Byte.Floor" and gives an empty value, a type with one number member should use the Number methods; `Color./` uses `Byte(Red / divisor)`, which floors.
- **A constant from a constant is typed by its name** (found 2026-10-20): `constant Window = Radius * 2 + 1` made `value / Window` typed "Radiu" (`IdentifierType` returns the name for an unknown identifier), BenchBlur used `constant Window = 9`.
- **A parameter named `index` inside a `for` is the parameter on the VM but the loop counter natively** (found 2026-10-20): `Averaged(pixels, width, height, index)` with a `for Window * Window` inside gave different results natively; either one rule for both or `strict check` rejects the name.
- **`constant Name ="..."` without a space compiles natively into a call of `Type.nstant`** (found 2026-10-20 in `Compiler/WebBuild.strict`): the Strict VM silently reads the constant as `""` (`constant Value =" a"` then `Value is " a"` fails with got ""), `strict check` reports 0 issues, `strict build` fails with "unsupported: call of Strict/Compiler/WebBuild.nstant" in the method using it. The parser should read it like `= "` or report the missing space with the file and line.
- **A `Generic` parameter of a non-generic type is not specialized** (found 2026-10-19, narrowed 2026-10-20): `Scored(model Generic)` calling `model.Area` or `Perplexity(model).Measured` fails on the Strict VM with "Unsupported native call Strict/Generic.Area" and `strict check` does not report it. A parameter of a generic type works (`Passed(scorer Perplexity)` takes `Perplexity(Gpt2.Tiny)` and `Perplexity(Qwen3.Tiny)`, VM and native, the earlier failure was an outdated `strict.exe`). Naming the specialization after the argument like `Specialization.Name` does for generic types (`Scored(Gpt2)`) passed its own tests but broke `Dictionary` and `List.Unique` (their `Generic` parameters mean the element type, also through `List(key Generic, mappedValue Generic)` and calls on the elements), so either the owner check must be exact or `strict check` should reject calls on such a parameter.
- **Big models fill the GPU and collect for seconds** (found 2026-10-19): Qwen3-Coder-30B-A3B (18.6 GB) leaves 3.6 GB of the RTX 4090, a few extra 831 token caches (`LookupDecoding` holds three) push it into a state where the GPU idles at 1% while every allocation collects. Done: the rows of a matrix are one unscanned allocation (`strict_packed_rows`, the marker skips words pointing into the large object it marked last), a full collection of the 96 expert model takes 0.23 instead of 1.17 s and it loads in 20 instead of 28 s. Done 2026-10-20: a list on the GPU whose count drops is freed at once (`strict_gpu_forget_resident`: block back to its class, residents entry and small download mark removed, 9099 lists in a 50 token Qwen3-0.6B lookup run, `STRICT_RC_CHECK` reports nothing new), Qwen3-0.6B lookup copying 1304-1430 instead of 307-334 tokens/s and prose 855-1011 instead of 337-429. Open: the residents table is searched linearly for that; a young collection still promotes the activations alive at that moment. What a full collection still marks are mostly the row lists built from the matrices (`Experts` keeps per expert `Joined` and `Stacked` lists next to the flattened ones, about 33M words for the 30B); blocks are still cut from 64 MB chunks that are never returned.
- **Qwen3-Coder-30B-A3B prefills 460 tokens at 2.0-2.4 ms per token** (found 2026-10-20, `Qwen3Benchmark ... 4 100 46`, the same before and after freeing GPU lists by count): the README has 0.17 ms per token for 800 tokens, Qwen3-8B prefills the same prompt at 0.15 ms; find where the grouped experts lose it (fewer than 64 tokens per expert take the flat experts?).
- **Unpacked rows of a matrix stay pinned** (2026-10-21): rows of `strict_packed_rows` live in an unscanned block, so a row unpacked on the CPU (`strict_list_unpacked`) pins its new holder (it was freed while used, Qwen3-0.6B crashed with `STRICT_GPU=0`); a model that is dropped keeps those holders. Better: decode packed rows where they are read instead of unpacking them, or make the block scanned once one of its rows is unpacked.
- **GPU kernels beyond NVIDIA** (found 2026-10-17, AMD chosen 2026-10-20): Done 2026-10-20: kernels generated from loops and the device runtime run on AMD through HIP (README), checked on an RX 480 (gfx803, Windows driver with HIP 5.2, which only loads code object v4): `Examples/Squares`, NeuralNetworks `Layer`, `Normalization`, `Softmax`, `Network` tests with `STRICT_GPU=all` and 3000 row dot products on warps of 32 lanes give the CPU results; the token and attention kernels run on AMD too, their warp operations gather operands from the other lanes (`ds_bpermute`, `NativeRuntimeAmd.mlir`): GPT-2 (`Gpt2Text`) writes the same text as on the CPU. Open: GPT-2 takes 48 ms per token on the RX 480 against 19.5 on the CPU (Ryzen 7 2700), each of the ~190 launches per token spends ~250 us on the host besides the 46 us HIP launch (graphs would save the launches, but HIP 5.2 on Windows runs out of memory after ~70 executable graphs, so AMD launches directly), find the host work per launch first (it is the same on NVIDIA); then `sdot4` from gfx906, WMMA on gfx11+, MFMA on CDNA instead of the gathered operands; the 512 thread tiles use 52 KB dynamic plus ~19 KB static shared memory, more than the 64 KB an AMD workgroup has, check what a Q4_K prefill does there; measure Qwen3-0.6B once it runs on this machine (see the next item); AMD on Linux (`libamdhip64.so`, ROCm) is untested; `hipFuncSetAttribute` takes host functions (only the 512 thread tiles call it). Tried: MLIR's `gpu` dialect in a `gpu.module` (`convert-gpu-to-nvvm` or `convert-to-llvm{dynamic=true}` with `nvvm-attach-target`) lowers ids, barriers and shuffles but routes `llvm.intr.floor` through libdevice (`__nv_floor`, ROCDL: `__ocml_floor`), so building would need the CUDA toolkit or ROCm; `gpu-module-to-binary` makes the same PTX as clang but takes 30 instead of 17 s for both targets. Vulkan needs kernels on memrefs (MLIR's SPIR-V conversion takes no LLVM dialect pointers) and a Vulkan host runtime, too big for now.

## Later GPU experiments (not blockers)
Measured on Qwen3-8B Q4_K_M and the RTX 4090, results in `NeuralNetworks/README.md`. Each optimization stays switchable (`STRICT_GPU=0`, `STRICT_GRAPHS=0`, `STRICT_YOUNG=0`, `STRICT_DP4A=0`) and gets measured before the next.
- Decode (6.8 ms per token, the GPU hardly idles; 2026-10-20 per step: products 5.33 ms at 780-920 GB/s, attention 0.55, the 72 rotations 0.23, norms 0.20, the 145 quantize kernels 0.15, gating, columns and sampling 0.13, so fusing all small kernels can win at most 18%; decode attention takes 14-18 us per layer at 200 positions with 8-16 blocks, Nsight Compute: 1900 instructions per warp at 12-13 cycles each, stalled on instruction fetch 31% (13000 SASS instructions, the unrolled merges and loads), memory 17%, barriers 11%, warm caches only 6% faster; Nsight Compute works here as Administrator): fewer of the 15 launches per layer (query and key rotations with the key and value append in one kernel, the 145 quantize kernels done by the kernels producing their inputs); the output projection, QKV and the 4 bit down projections reach 775-790 GB/s against 885 for gate and up. RTX 3080 (Qwen3-0.6B Q4_K_M, 2.3 ms per token): per layer the 4 products take 26 us, attention 17.5 us (latency, not memory: splitting the positions every 32 or 64 instead of 128 changes nothing, needs Nsight Compute, which needs the GPU performance counters enabled for all users), the 2 rotations 9 us, the 2 norms 6.7 us, the 4 quantize kernels 4.8 us; starting graph segments at 32 or 128 instead of 8 changes nothing (the GPU is the limit); quantizing inside the dot products costs a Q6_K lane 4 groups of 32 floats for 16 numbers, so the producing kernels should write the 8 bit numbers.
- Prompt (0.15 ms per token at 460 tokens, llama.cpp 0.10): the GPU idles between the 545 launches of a 460 token prompt and when a full collection lands in a prefill; tiles of 128 by 128 with 8 warps like llama.cpp; Q6_K rows are not unpacked in the tiles, 8 bit rows still take `strict_device_blocks`; one threshold for strided sums like `Merged` for all loops; GPT-2 prefills on the CPU.
- Batches: sequences of different lengths and continuous batching, peak VRAM in `Qwen3Benchmark`.
- Long context: the attention kernel uses 255 registers, batch 32 runs its blocks in two waves.
- Qwen3-Coder-30B-A3B launches 1157 kernels per token and its first upload of the 18.6 GB of experts is bound by the CPU copy (a pinned double buffered upload made the copies 13 times faster on the GPU but not the first token); GPT-2 captures its launches again after every download (5 graph segments per token, the main thread is in the CUDA driver); per row `Summed` still checks its lists on the CPU; MXFP4 for gpt-oss; paged caches, prefix caching and inactive caches on the CPU for bigger models.

## Base library (found 2026-10-19 comparing DeltaEngine, AiInvest, old SoulCraft, .NET, std, Go and Rust)
- **Not added yet** (chosen 2026-10-19 to wait): Number `Sign`, `Wrap`, inverse trigonometry (`ArcTangent2` has 53 uses in the C# DeltaEngine), Text to Numbers, fixed decimal number text, `Truncated`.

## Deferred
- **Qwen3-Coder-Next 80B-A3B pruned to 18-20 GB** (deferred 2026-10-19 until there is more VRAM): its 512 experts are 77.3 of its 79.7B weights, at Q4_K only 176 of them fit into 18.7 GB. llama.cpp (8 chunks of 512 tokens, English and Strict perplexity): all experts 1.14 and 2.08, the 176 experts of most activation energy 1.92 and 5.81 with 8K calibration tokens or 3.24 and 2.33 with 48K, against Qwen3-Coder-30B-A3B with 96 of 128 experts 1.47 and 2.16 (14.2 GB). About 256 experts would need 3 bit experts, not worth it now. Needs the `qwen3next` architecture in Strict (Gated DeltaNet, shared with Qwen3.8 below, 512 experts using 10 plus a shared expert with a sigmoid gate).

## Robot arm voice (added 2026-10-19)
- **Questions to the arm answered by Qwen3.8 on this server, spoken back.** The robot arm now has a microphone and a speaker. Short known answers play locally on the arm, anything complex goes to Qwen3.8 running here, the answer text becomes speech with [Fish Audio Drama 3](https://fish.audio/drama-3/) (its API with the model header `drama-3`, directions for the delivery written into the text) and is played on the arm. Qwen3.8 is `qwen3.8:27b` in Ollama (GGUF architecture `qwen35`, 16.8 GB at Q4_K): 64 layers of width 5120, every fourth full attention (24 heads of 256, 4 key and value heads, rotary on 64 of the 256), the others Gated DeltaNet (16 key heads, 48 value heads of 128, a 128 by 128 state each, a convolution of 4), plus one MTP layer that drafts the next token (HyperQwen's drafter). Done 2026-10-21: `Qwen3.Loaded` runs it (`DeltaBlock`, `Hybrid`, `HeadRows`, the same 24 tokens as Ollama, 22.1-22.5 ms per token on the RTX 4090, Ollama 21.5). Open: the prefill still updates the memory token by token (3 launches per token and layer, 100 tokens 4.8 ms per token, a chunked delta rule would use matrix products), batches (`Copies`) and lookup speculation (`Truncated` cannot cut a recurrent memory, the drafts need the memory before them), the MTP layer as the draft model of `Speculation`, speech to text for the questions (on the arm or here), the decision what stays local, a small endpoint in Strict the arm calls, and the time from the end of a question to the first sound.

## Web, macOS, Android and iOS (added 2026-10-20)
All DeltaEngine games (`../DeltaEngine/Games`) playable in the browser on the website (advertising for Strict and the Steam/store games), later Steam (Windows and Linux/Steam Deck exist), Android/iOS for SoulCraft, macOS. Only the output runs there, not Strict itself. Steps in order, each a small demo first: hello world in the browser (done: `strict build Examples/HelloLogger.strict web`, wasm32: the C functions with sizes go through 32 bit wrappers in `NativeRuntimeWeb.mlir`, `WebBuild.Sized`), an empty WebGL window (done: `Games/Samples/ClearColor`, Asyncify continues `Window_Frame` at the next animation frame), drawing (done: `ShapesDemo`, the collector finds values in wasm locals through `emscripten_scan_registers`), textures and text (done: `TextDemo`, `AtlasDemo`, the `Content` folder next to the program is preloaded as `/Content`), the page `Compiler/WebShell.html`, sound and input (done: `LogoApp`, the emscripten-glfw port for gamepads), the first games (done: Artifex, WebWars: the collector reads every 4 bytes as a possible reference in wasm32, `Window_Frame` waits for the next animation frame). Also done: builds 5-10 times faster (the `.ll` compiled at the build level, linked at -O1, wasm-opt without coalesce-locals and without inlining big functions with one caller: WebWars 300 to 27 s, ArenaWars 69 s; the inlined 2.5 MB `Run` of ArenaWars made V8 use over 20 GB and the tab crash), function names in the wasm for browser profiles, GL errors checked once per frame (every `glGetError` waits for the GPU process), touch as mouse and DXT decoded without s3tc for mobile browsers. All games and samples build and run in headless Chrome (`Games/*` at 60 fps, RocketCommander 40, Breakout 49). Open: a test on real phones (Android Chrome, iOS Safari; the wasm memory starts at 1 GB for the Strict heap, likely too much for iOS), download sizes (SoulCraft 150 MB, RacingGame 51 MB, ArenaWars 46 MB of `.data`: compression on the server, smaller textures for phones), file names without content hashes (a browser can combine a cached old `.wasm` with a new `.js` after an update, seen locally with CombatDemo), sound now that the page starts without a click (browsers keep audio suspended until the first input), saves and settings on IDBFS, why RocketCommander and Breakout stay below 60 fps. Old reference: `C:\code\DeltaEngine\DeltaEngine.Javascript`, `C:\code\DeltaEngine\Services\DeltaEngineServices.Platforms.Web` and the C# WebGL framework in `C:\code\GitHub\DeltaEngine\DeltaEngineFrameworks` (browser window, gamepad and fullscreen APIs, local storage settings, async content loading).
- **Target in the compiler**: `Web` next to Windows/Linux/MacOS in `Compiler/Platform.strict` (later Android, IOS) with its triple (`wasm32-unknown-emscripten`, wasm32 since memory64 is not in every browser), no `-march=native`; `strict build Game.strict web`. `Toolchain` finds or installs emsdk like LLVM: our clang 22 compiles the `.ll` (the IR matches its LLVM), `emcc` compiles the `native/` C files with `native/Web.flags` and links (`-sUSE_GLFW=3`, WebGL 2, memory growth up to 4 GB, Asyncify, `--preload-file` of the content, an HTML shell); `Release/Web/` gets `index.html`, `.js`, `.wasm`, `.data`.
- **Runtime** (`NativeRuntimeWeb.mlir` and target dependent parts of `NativeRuntime.mlir`): `size_t` is i64 in the libc declarations (a type per target), the 64 GB reserve and 128 MB card table scale with the target (chunks with memory growth on the web), GC roots via `emscripten_scan_stack` and `emscripten_scan_registers` (need Asyncify, like Boehm GC on Emscripten) instead of reading the native stack, no signal/VEH guard, no threads (`strict_pool_spawn` returns 0, `llvm.x86.sse2.pause` only on x86), no GPU, processes and libraries give an Error; files, clock and directories through Emscripten's musl (struct offsets of musl wasm32, not glibc x86_64). Console programs and the base type tests run in node with the same output as natively.
- **Main loop**: the games keep their blocking loops, `Window_Frame` yields with `emscripten_sleep(0)` through Asyncify instead of `glFinish`; frame callbacks or JSPI only if Asyncify costs too much.
- **Engine natives** (`../DeltaEngine`): `Web.flags` without the GLFW sources and glad, miniaudio with Web Audio; one GLES helper for `Render.c` and `Gpu.c` turns `#version 330 core` into `#version 300 es` with precisions (also for Android/iOS), `Gpu.c` calls GLES3 directly instead of `wglGetProcAddress`/`dlsym`; DDS/DXT only with `WEBGL_compressed_texture_s3tc`, otherwise the content build converts to png; audio starts after a click ("Click to play" also gives keyboard focus and fullscreen); gamepads through GLFW's Gamepad API; content preloaded so synchronous reads keep working, saves and settings on IDBFS; `GoogleSheet.c` not on the web; shell with canvas, HiDPI, loading progress, fullscreen, links to the store games.
- **Games** in order: Samples and Breakout, the 2D games (WebWars, TinyBat, Blocks, GhostWars, Artifex), SoulCraft (content size, touch as mouse on mobile browsers), the 3D games (RocketCommander, XnaRacer, ArenaWars: shaders, texture arrays, DDS). Per game: download size, memory, keys browsers keep (ESC leaves fullscreen), sound after the first click, frame time against the native exe. The legacy C#/XNA games need a Strict port first.
- **Later platforms**: macOS (done: Intel tested, Apple Silicon cross-built but not run yet, GLFW cocoa windows (`MacOS.flags` in DeltaEngine, LogoApp runs), `.dylib` libraries, `.app` bundles signed with a Developer ID or ad hoc; open: run and test on Apple Silicon, notarization (`xcrun notarytool submit` and `stapler` need an Apple developer account), GPU kernels through Metal: barely working, see README; open: shared memory (threadgroup arrays) for the summing and grouped kernels, a real allocator instead of one bump buffer that only resets when everything is freed, launches without waiting before every copy, simdgroup_matrix versions of the tensor core kernels, measuring on Apple Silicon), Android (NDK `aarch64-linux-android`, NativeActivity instead of GLFW, EGL and GLES3, miniaudio AAudio, touch, content from APK assets, AAB and signing), iOS (`arm64-apple-ios`, a small UIKit shim, GLES3 or ANGLE on Metal, needs a Mac and Xcode), Steam (Steamworks SDK as a small native package: achievements, cloud saves, overlay).
- **`strict test NeuralNetworks lines` crashes natively** (found 2026-10-20 on macOS, every file passes on its own): after the Perceptron tests the history walk of `strict_list_restoring` reaches a null link before the live items (also with `STRICT_RC=0` and `STRICT_YOUNG=0`), likely a shared constant like `Gpt2.Tiny` used again after other files appended to its lists in place.
- **Implicit self calls in replacement constructors fail natively** (found 2026-10-20): `Creature(value, previous = Pose(9))` gives "unsupported: call of Creature.Pose in Creature_Moved_2", `value.Pose(9)` works.
