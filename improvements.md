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

2. **Values rewritten in place** (most important, decided 2026-10-19: game logic, rendering, entities, physics and neural networks update the same values over and over). Done: lists built by literals, constructors and collecting `for` loops append in place, a local that only ever holds fresh values and never leaks into another local (`MlirOwned`) frees its old value when it is overwritten, native builds inline small methods, flat Vector3 (10M `position = position + velocity` 793 to 100 ms, Digits28 546 to 465 ms), and a struct construction stored back into such a local (a replacement constructor `entity = Mover(entity, position = entity.position + entity.velocity)`) writes only its changed slots into the old block when `strict_rc_unshared` finds nobody else holding it, a flat result of a call stored into such a local is built into the dying old block (destination passing, `MlirDestined`, 10M `position = Moved(position)` 793 to 84 ms), and a collecting `for` mapping a list to a list of the same type refills the dying list, building each flat element into the old one (`MlirListDestined`, 1000 frames of 10k `value + velocity` 474 to 225 ms). A local whose stores are all constructions and whose reads only load fields (`MlirAlone`) is proven alone at compile time, its reuse only checks for an empty slot (10M entity updates 134 to 94 ms). Calls returning a number, boolean or text and inlined aliases read only like fields keep a local proven alone. In the Strict VM a frame writes its slots in place now (`ErrorCheck` takes the two values it checks instead of the slot list, so `Frame.Finished` owns its copy and writes with `strict_list_set` instead of a versioned write per instruction) and a constructor given exactly the count its `with Length is N` member needs skips `Validated` (also natively), and a struct a method constructs and returns is rebuilt in the dying block its caller hands it (`MlirStructs`, struct destinations through up to two returning calls, so `VmValue` results of the VM reuse the old values of locals), 100k Vector3 updates 1.07 to 0.60 s. A flat result stored into an element of an owned list outside a collecting loop is built into the old element (`MlirElements`, an inline block when nothing holds the list, a separate one when only the list does; 10M `positions(0) = positions(0) + velocity` 250 to 175 ms), and a member of a struct rebuilt in place into its old block (`MlirMembered`, 10M Vector3 member updates 230 to 103 ms).
3. **Flat component lists** (most important long term, decided 2026-10-19; done: `value.Bits is 32` on the members of the Math types, natively Vector2 and Size are one 64-bit slot of two 32 bit floats, unpacked where a list is expected and packed where a list is given as a Vector2, see `Compiler/README.md`, one-number types (Degrees, Radians, Ratio) one 64-bit slot rounded to 32 bit when constructed, like the Strict VM rounds such members, a type of 3 to 16 numbers whose methods only read `X`, `Y`, `Z`, `W`, `Length` or convert them to text (Vector3) is one block of its count and 32 bit floats, `MlirFlats`, loading its list from outside converts it). Entities keep their components in huge lists per component type (all sprites share one list of their render data, like `RenderDataArray` in the C# DeltaEngine `c:/code/DeltaEngine/DeltaEngine.Entities`), components are pure data and most only hold numbers. A list of a flat type built by a collecting loop keeps its elements in one block after its items (`strict_flat_list_new`), constructions and destined calls build the next element in place (`strict_flat_slot`, 1000 frames of 10k Vector3 225 to 131 ms, no counts per element anymore). Matrix and Quaternion are flat too: another instance's member that is only indexed (`other.values(4)`) is read in place instead of converted (1M matrix products 6.6 s to 270 ms), arithmetic on the member converts it (`MlirPacking`, Quaternion Lerp 255 to 200 ms, natively the same 32 bit results as the Strict VM). Open: loops still read each element of a flat list through its item slot, walking the inline blocks directly would allow SIMD and GPU uploads without packing.

## Language and compiler
- **GPU kernels beyond NVIDIA** (found 2026-10-17): the device runtime and the tensor core kernels (`NativeRuntimeTokens.mlir`, `NativeRuntimeAttention.mlir`) call NVVM intrinsics directly. MLIR 22 lowers `gpu.subgroup_mma_*` to NVVM (wmma) but not to ROCDL ("failed to legalize gpu.subgroup_mma_load_matrix"), SPIR-V needs `spirv.entry_point_abi`. Before writing the next kernel by hand, check what MLIR already provides (`linalg` with the transform dialect's tensor core tiling, `nvgpu`/`amdgpu` mma ops), then port the existing kernels to it so AMD (ROCDL) and Vulkan (SPIR-V) get them too.

## Later GPU experiments (not blockers)
- A CPU loop over the packed rows of a matrix unpacks them to doubles in place (found 2026-10-19): the shared matrix stays 4 times larger and later loops over it (below 4M numbers, no longer resident) stay on the CPU, GPT-2 evaluation went from 5 to 35 ms per token after `Activity` read MLP activations on the CPU (`NeuronPruning` now loads a second model to measure).
Measured on Qwen3-8B Q4_K_M and the RTX 4090, results in `NeuralNetworks/README.md`. Each optimization stays switchable (`STRICT_GPU=0`, `STRICT_GRAPHS=0`, `STRICT_YOUNG=0`, `STRICT_DP4A=0`) and gets measured before the next.
- Decode (6.6 ms per token, the GPU hardly idles): fewer of the 15 launches per layer (query and key rotations with the key and value append in one kernel, the 145 quantize kernels done by the kernels producing their inputs); the output projection, QKV and the 4 bit down projections reach 775-790 GB/s against 885 for gate and up.
- Prompt (0.15 ms per token at 460 tokens, llama.cpp 0.10): the GPU idles between the 545 launches of a 460 token prompt and when a full collection lands in a prefill; tiles of 128 by 128 with 8 warps like llama.cpp; Q6_K rows are not unpacked in the tiles, 8 bit rows still take `strict_device_blocks`; one threshold for strided sums like `Merged` for all loops; GPT-2 prefills on the CPU.
- Batches: sequences of different lengths and continuous batching, peak VRAM in `Qwen3Benchmark`.
- Long context: the attention kernel uses 255 registers, batch 32 runs its blocks in two waves.
- Qwen3-Coder-30B-A3B launches 1157 kernels per token and its first upload of the 18.6 GB of experts is bound by the CPU copy (a pinned double buffered upload made the copies 13 times faster on the GPU but not the first token); GPT-2 captures its launches again after every download (5 graph segments per token, the main thread is in the CUDA driver); per row `Summed` still checks its lists on the CPU; matrix rows are separate allocations (a full collection marks the 1.7M rows of the 8B, 140-216 ms); MXFP4 for gpt-oss; paged caches, prefix caching and inactive caches on the CPU for bigger models.

## Base library (found 2026-10-19 comparing DeltaEngine, AiInvest, old SoulCraft, .NET, std, Go and Rust)
- **Ceilings**: `MovingWindow.Minimum` and `Maximum` scan the window (a monotonic queue would make them O(1)), `Sorted` needs `<` on the elements (Text has none yet).
- **Not added yet** (chosen 2026-10-19 to wait): Number `Sign`, `Wrap`, inverse trigonometry (`ArcTangent2` has 53 uses in the C# DeltaEngine), Text to Numbers, fixed decimal number text, `Truncated`; `NeuralNetworks/Optimizer.Power` duplicates `^`.
