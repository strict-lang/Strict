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
      3.7 and prose 2.7 times). Open: the products with the weights for 2 to 16 tokens
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
      Qwen3.8 is a hybrid of Gated DeltaNet and attention layers, `Qwen3` cannot load it yet.

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

## Language and compiler
- **A `Generic` parameter of a non-generic type is not specialized** (found 2026-10-19, narrowed 2026-10-20): `Scored(model Generic)` calling `model.Area` or `Perplexity(model).Measured` fails on the Strict VM with "Unsupported native call Strict/Generic.Area" and `strict check` does not report it. A parameter of a generic type works (`Passed(scorer Perplexity)` takes `Perplexity(Gpt2.Tiny)` and `Perplexity(Qwen3.Tiny)`, VM and native, the earlier failure was an outdated `strict.exe`). Naming the specialization after the argument like `Specialization.Name` does for generic types (`Scored(Gpt2)`) passed its own tests but broke `Dictionary` and `List.Unique` (their `Generic` parameters mean the element type, also through `List(key Generic, mappedValue Generic)` and calls on the elements), so either the owner check must be exact or `strict check` should reject calls on such a parameter.
- **A replacement constructor of a block holding the newest cache costs 30%** (found 2026-10-19): `updated.Add(Qwen3Block(cached, queries = List(Number)))` in `Qwen3.Steps` (to drop the queries of the last step, 650 MB after an 831 token prefill of Qwen3-Coder-30B) made Qwen3-8B lookup decoding 400 instead of 594 tokens/s, as if the keys and values were copied or no longer appended in place afterwards; `Qwen3.Steps` keeps `updated.Add(cached)` until this is understood.
- **Big models fill the GPU and collect for seconds** (found 2026-10-19): Qwen3-Coder-30B-A3B (18.6 GB) leaves 3.6 GB of the RTX 4090, a few extra 831 token caches (`LookupDecoding` holds three) push it into a state where the GPU idles at 1% while every allocation collects. Done: the rows of a matrix are one unscanned allocation (`strict_packed_rows`, the marker skips words pointing into the large object it marked last), a full collection of the 96 expert model takes 0.23 instead of 1.17 s and it loads in 20 instead of 28 s. Open: reference counting leaves every list on the GPU to the collector (`strict_rc_device_bound`), a young collection promotes the activations alive at that moment and their device blocks only come back with a full collection, so an 800 token prefill of the 30B still runs about one (214 ms, GPU blocks cut since the last full collection pass half of the free memory). Freeing a list on the GPU when its count drops would need: its block back to its class (the holder keeps device address and class, later launches on the one stream run after the ones reading it), its entry in the residents table removed (a linear search now) and its segment mark for small downloads cleared. What a full collection still marks are mostly the row lists built from the matrices (`Experts` keeps per expert `Joined` and `Stacked` lists next to the flattened ones, about 33M words for the 30B); blocks are still cut from 64 MB chunks that are never returned.
- **GPU kernels beyond NVIDIA** (found 2026-10-17): the device runtime and the tensor core kernels (`NativeRuntimeTokens.mlir`, `NativeRuntimeAttention.mlir`) call NVVM intrinsics directly. MLIR 22 lowers `gpu.subgroup_mma_*` to NVVM (wmma) but not to ROCDL ("failed to legalize gpu.subgroup_mma_load_matrix"), SPIR-V needs `spirv.entry_point_abi`. Before writing the next kernel by hand, check what MLIR already provides (`linalg` with the transform dialect's tensor core tiling, `nvgpu`/`amdgpu` mma ops), then port the existing kernels to it so AMD (ROCDL) and Vulkan (SPIR-V) get them too.

## Later GPU experiments (not blockers)
Measured on Qwen3-8B Q4_K_M and the RTX 4090, results in `NeuralNetworks/README.md`. Each optimization stays switchable (`STRICT_GPU=0`, `STRICT_GRAPHS=0`, `STRICT_YOUNG=0`, `STRICT_DP4A=0`) and gets measured before the next.
- Decode (6.6 ms per token, the GPU hardly idles): fewer of the 15 launches per layer (query and key rotations with the key and value append in one kernel, the 145 quantize kernels done by the kernels producing their inputs); the output projection, QKV and the 4 bit down projections reach 775-790 GB/s against 885 for gate and up. RTX 3080 (Qwen3-0.6B Q4_K_M, 2.3 ms per token): per layer the 4 products take 26 us, attention 17.5 us (latency, not memory: splitting the positions every 32 or 64 instead of 128 changes nothing, needs Nsight Compute, which needs the GPU performance counters enabled for all users), the 2 rotations 9 us, the 2 norms 6.7 us, the 4 quantize kernels 4.8 us; starting graph segments at 32 or 128 instead of 8 changes nothing (the GPU is the limit); quantizing inside the dot products costs a Q6_K lane 4 groups of 32 floats for 16 numbers, so the producing kernels should write the 8 bit numbers.
- Prompt (0.15 ms per token at 460 tokens, llama.cpp 0.10): the GPU idles between the 545 launches of a 460 token prompt and when a full collection lands in a prefill; tiles of 128 by 128 with 8 warps like llama.cpp; Q6_K rows are not unpacked in the tiles, 8 bit rows still take `strict_device_blocks`; one threshold for strided sums like `Merged` for all loops; GPT-2 prefills on the CPU.
- Batches: sequences of different lengths and continuous batching, peak VRAM in `Qwen3Benchmark`.
- Long context: the attention kernel uses 255 registers, batch 32 runs its blocks in two waves.
- Qwen3-Coder-30B-A3B launches 1157 kernels per token and its first upload of the 18.6 GB of experts is bound by the CPU copy (a pinned double buffered upload made the copies 13 times faster on the GPU but not the first token); GPT-2 captures its launches again after every download (5 graph segments per token, the main thread is in the CUDA driver); per row `Summed` still checks its lists on the CPU; MXFP4 for gpt-oss; paged caches, prefix caching and inactive caches on the CPU for bigger models.

## Base library (found 2026-10-19 comparing DeltaEngine, AiInvest, old SoulCraft, .NET, std, Go and Rust)
- **Not added yet** (chosen 2026-10-19 to wait): Number `Sign`, `Wrap`, inverse trigonometry (`ArcTangent2` has 53 uses in the C# DeltaEngine), Text to Numbers, fixed decimal number text, `Truncated`.

## Deferred
- **Qwen3-Coder-Next 80B-A3B pruned to 18-20 GB** (deferred 2026-10-19 until there is more VRAM): its 512 experts are 77.3 of its 79.7B weights, at Q4_K only 176 of them fit into 18.7 GB. llama.cpp (8 chunks of 512 tokens, English and Strict perplexity): all experts 1.14 and 2.08, the 176 experts of most activation energy 1.92 and 5.81 with 8K calibration tokens or 3.24 and 2.33 with 48K, against Qwen3-Coder-30B-A3B with 96 of 128 experts 1.47 and 2.16 (14.2 GB). About 256 experts would need 3 bit experts, not worth it now. Needs the `qwen3next` architecture in Strict (Gated DeltaNet, shared with Qwen3.8 below, 512 experts using 10 plus a shared expert with a sigmoid gate).

## Robot arm voice (added 2026-10-19)
- **Questions to the arm answered by Qwen3.8 on this server, spoken back.** The robot arm now has a microphone and a speaker. Short known answers play locally on the arm, anything complex goes to Qwen3.8 running here, the answer text becomes speech with [Fish Audio Drama 3](https://fish.audio/drama-3/) (its API with the model header `drama-3`, directions for the delivery written into the text) and is played on the arm. Qwen3.8 is `qwen3.8:27b` in Ollama (GGUF architecture `qwen35`, 16.8 GB at Q4_K): 64 layers of width 5120, every fourth full attention (24 heads of 256, 4 key and value heads, rotary on 64 of the 256), the others Gated DeltaNet (16 key heads, 48 value heads of 128, a 128 by 128 state each, a convolution of 4), plus one MTP layer that drafts the next token (HyperQwen's drafter). Open: Gated DeltaNet and the attention variant in Strict (a sequential prefill first, a chunked one later), the MTP layer as the draft model of `Speculation`, speech to text for the questions (on the arm or here), the decision what stays local, a small endpoint in Strict the arm calls, and the time from the end of a question to the first sound.
