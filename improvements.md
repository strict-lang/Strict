# Improvements
In priority order. Remove what is done, add what is found, keep each item to what is still open (results go into the README of the package). Every item gets done before the bigger image processing and neural network examples, parallel loops and GPU (decided 2026-10-12).

1. **Decode speed** (Qwen3-8B Q4_K_M on the RTX 4090 at 6.6-6.7 ms per token, 150 tokens/s, results in `NeuralNetworks/README.md`): a step is now its kernels (6.55 ms), the GPU hardly idles. Open: fewer of the 15 launches per layer (attention is one kernel now): the query and key rotations in one kernel with keys and values appended as one contiguous part, the 145 quantize kernels (1 us each) done by the kernels that produce their inputs; the matrix products reach 885 GB/s for gate and up but 775-790 for the output projection, QKV and the 4 bit down projections (2-4 us of each kernel are ramp-up and tail).
2. **Prompt speed** (Qwen3-8B Q4_K_M on the RTX 4090, what is done and measured is in `NeuralNetworks/README.md`): prefill takes 1.0-1.2 ms per token at 10 tokens, 0.21 ms at 100, 0.15 ms at 460, 0.15 ms at 1820 and 0.17 ms at 3720 (the fastest of 3 prefills in `Qwen3Benchmark`; the matrix products of a prompt are tiles of 128 rows by 64 tokens, `strict_device_tiles`, which add the biases when they store (with fewer tiles than multiprocessors the blocks of a tile split its columns and the last adds up their sums, 70 and 100 tokens 16% faster), `Qwen3.Prefilled` sends up to 2048 tokens at once) (llama.cpp 1.18, 0.175 and 0.10 ms). Open: the GPU idles between the 545 launches of a 460 token prompt and when a full collection lands in the prefill (81 ms of kernels in 200 to 280 ms); the generated kernels reading exact positions (`Rotated`, `Normalized`, `Experts.Gated`, 38 of 291 ms at 1820 tokens) are mostly bound by memory: without the float fallback of a read they execute 14-20% fewer instructions but save only 2 ms (and a non-integral position would fail instead of being cut like on the Strict VM), exact fractions of floats (power of 2 denominators) instead of the fallback make the denominators of parameters unknown and cost more than they save, branching around the fallback did not move the float positions out of the way; the tiles take mma.m16n8k32 from sm_80 on and load the next 256 columns into registers while computing, ncu at 460 tokens (gate and up, base clocks, 746 instead of 1022 us; 128 instead of 64 rows per block 807 to 763 us, the token sums added up when the tokens are put 749): 32 instructions per m16n8k32 (16 of them the float scaling after every 32 numbers), its 8 steps of 32 numbers take 44% of the time, staging the next 256 columns 36% (setup and the first stage 10%, prepare 12%, put 9%, fetch 5%), the barriers 14%, the stores 6%; decoding the scales of the Q4_K headers in put instead of prepare only moved the work (more instructions, same time), blocks that go on to their next tile (one per SM, the first stage of a tile loaded during the last products of the one before) took 880 instead of 749 us (spills, more state per 256 columns). Next: tiles of 128 by 128 with 8 warps of up to 255 registers like llama.cpp (each warp 32 by 64, more products per staged byte and per scale load); accumulating a block of 256 in integers needs one token scale per 256 (like Q8_K) and 16 more registers; Q6_K rows are not unpacked yet (9.9 instead of 7.4 us per million weights), 8 bit rows still take `strict_device_blocks`; prefill times jump when a full collection lands in them (910 tokens 0.41-0.49 ms); the 32768 outputs from which a strided sum like `Merged` takes a thread per output are one threshold for all loops, measure it per loop when another kernel needs a different one; GPT-2 gets the same prefill (its prompt runs on the CPU, no loop reads 4M numbers before the logits).
3. **Batched decoding** (done and measured in `NeuralNetworks/README.md`): batch 8 decodes 953 tokens/s (llama.cpp 727) and 32 sequences 2357 tokens/s on the tensor cores (`strict_device_blocks`, 128 registers because the row by row path for other rows is in the same kernel, so 2 blocks of 8 warps per SM, capped at 80 or 64 registers it spills and gets slower); the Q6_K rows reach 540 GB/s against 790 for Q4_K at batch 8. Open: sequences of different lengths and continuous batching (all sequences are at the same position now), peak VRAM in `Qwen3Benchmark` (p95 is reported; batch 8 decodes 525 tokens/s after 3720 tokens with the 8 bit cache, median 15.2 ms per step, p95 51 ms because the first steps are slower).
4. **Long context decoding** (8 bit cache and flash decoding done, results in `NeuralNetworks/README.md`): decoding takes 7.6 ms after 1820 and 8.3 ms after 3720 tokens, 8.6 ms after 7450, 9.7 ms after 14900 and 12.8 ms after 29700 tokens with the 8 bit cache (prefill 0.19 and 0.25 ms per token). Open: the attention kernel uses 255 registers, so a block of 8 warps per SM and batch 32 runs its 256 blocks in two waves (13.8 ms per step).
5. **Language model leftovers**: Qwen3-Coder-30B-A3B launches 1157 kernels per token, which keeps the CPU busy, and `Qwen3CoderText` reports 430 ms per token because its 24 tokens include the first upload of the 18.6 GB of experts (283 synchronous copies at 2.3 GB/s from pageable memory; a pinned upload, two 8 MB chunks on a stream of their own, made the copies of Qwen3-0.6B 13 times faster on the GPU, 1105 to 81 ms, but not its first token, copying into pinned memory is the same CPU work as the staging of the driver); GPT-2 takes 1.3-1.8 ms per token under nsys with 0.72 ms of kernels, after every download the launches are captured again in segments of 8, 16, 32, 64 and the rest (5 graphs per token, each captured, updated, launched and destroyed) and the main thread spends its time in the CUDA driver, not in Strict code; MXFP4 for gpt-oss, safetensors other than F32 (`OnlyFloat32TensorsAreSupported`); per row `Summed` still checks its lists on the CPU, matrix rows are separate allocations (a full collection marks the 1.7M rows of the 8B, 140-216 ms); later for bigger models paged caches, prefix caching and moving inactive caches to the CPU. Each optimization stays switchable (`STRICT_GPU=0` for the CPU, `STRICT_GRAPHS=0`, `STRICT_YOUNG=0`, `STRICT_DP4A=0`) and gets measured before the next.
6. **Memory reuse and neural networks** (results in `NeuralNetworks/README.md` and the README memory model): objects die by their reference counts (Digits28 834 to 520 ms). Open: (a) a unique value (count 0, referenced by one local only) rewritten in place instead of copied, like `Layer.Moved` building new rows from the old ones; (b) objects below 256 bytes that hold no reference to a counted one stay with the collector (freeing each one made strict.exe building itself 64 instead of 51 s), freeing them in bulk per page could change that; (c) bigger networks to measure parallel `for` on all cores, SIMD and the GPU against the C# version (only small inline data here, bigger samples get their own repository like `../Mnist`).
7. **Flat number types** (decided 2026-10-18, done: `value.Bits is 32` on the members of the Math types, natively Vector2 and Size are one 64-bit slot of two 32 bit floats, unpacked where a list is expected and packed where a list is given as a Vector2, see `Compiler/README.md`, one-number types (Degrees, Radians, Ratio) one 64-bit slot rounded to 32 bit when constructed, like the Strict VM rounds such members). Open: (a) types of 3 to 16 numbers (Vector3, Quaternion, Rectangle, Ray, Matrix) as one block of exactly their 32 bit floats without a list header, a list of them one array of such elements; (b) assigning a replacement or `position + offset` back to the same mutable variable could write in place.
8. **Tooling**: `DroppedResultMustBeUsed` knows the methods of the checked folder and the base types (see `Validators/README.md`), methods of other packages need their bodies (or a never fails flag in the bytecode) in the checks. `strict test <folder> lines` builds all files of the folder in one executable (Bytecode: 44 files in 50 s, about 40 s of it clang). Copies of other packages' types stay in `.strictbinary` files (references measured 2026-10-18: checking fresh binaries 2.1 times slower for 3% smaller files). The LanguageServer and the Debug Adapter Protocol for the VM wait until after the parallel loops and GPU (decided 2026-10-13).
9. **Reflection** (only as use cases need it): `to Type` of a Text that is not a literal (needs type names at runtime), natively a `Variable` value is a `Generic` without its type, a `Stacktrace`'s `Method` is `Type.Method` text (a `Method` value needs the methods at runtime).

## Research (when all other work is done)

10. **Specialized language models: pruning and parallel generation**

    Goal: extreme measured speed and small resident models for English text and Strict code.
    Accept some accuracy loss and loss of unrelated languages, tasks and general knowledge,
    provided held-out English and Strict generation remain useful. Prefer parallel execution;
    optimize time to useful output and passing code, not only raw tokens/s.
    Results belong in `NeuralNetworks/README.md`; keep this entry to remaining experiments.

    - **Baseline and measurement first.** Start with GPT-2 and Qwen3-0.6B, available locally.
      Use a tiny trained model to check compression mechanics; the existing tiny fixtures only
      check inference correctness. Do not infer useful generation quality from those fixtures.
      Keep separate calibration/training and held-out evaluation data: English plus Strict
      source, with unseen coding tasks and executable assertions. Record original-model results
      before changing weights. GPT-2 can test mechanics even if its Strict coding baseline is weak.
      - Implement stable next-token log-probabilities and held-out perplexity, using GPU scoring
        and reductions where practical. Keep prompts, tokenization and scored positions fixed.
        If tokenization changes, compare negative log-likelihood per source byte (bits/byte),
        not token perplexity across different vocabularies; record tokens per source byte too.
      - Measure weight-file bytes, parameter count, peak RAM/VRAM, cold load/time to first token,
        warmed prefill and decode, median/p95 latency, tokens/s and generated bytes/s.
        Report single-sequence latency separately from aggregate batched throughput.
      - Judge Strict by `strict check`, executable tests and elapsed time until the whole task
        passes, including planning, prefill, retries and repairs. Track English coherence and
        task completion too. Publish the speed/quality curve and choose its useful tradeoffs;
        do not silently change evaluation data or claim unchanged quality from a few samples.
      - GPU agent reports a lossless upload-format fix: CPU embedding reads had widened rows
        to doubles, causing tied embedding/output matrices to upload at 64 bits. Rows now use
        the other rows' packed format only when every value is exactly representable, otherwise
        retain the wide format. Reported matrix uploads: GPT-2 480 to 249 MB, Qwen3-0.6B 1919
        to 849 MB; generated text unchanged. Reported GPT-2 decode is now about 1.4 ms/token
        (1.3-1.8 ms profiled), including 0.72 ms GPU work; remaining overhead includes driver
        re-recording of five graph segments. Re-measure these baselines before pruning; this
        storage correction is not pruning. Pinned staging was reverted: faster GPU copy alone
        did not improve time to first token. Measure end-to-end gains, not isolated copy timings.

    - **Whole-layer pruning first.** Measure each block's effect on hidden states over calibration
      inputs, then confirm candidate removals with held-out perplexity and generation tests.
      Similar hidden states are a ranking heuristic, not proof that a block is dispensable.
      Remove one block at a time and re-rank; test progressively larger reductions and physically
      remove weights and caches. Measure saved launches, memory traffic and actual decode time.
      Small models may tolerate less removal. Try approximate layer merging only if deletion
      establishes a useful baseline; nonlinear transformer blocks cannot simply be multiplied
      together. Reference: [ShortGPT](https://arxiv.org/abs/2403.03853).

    - **MLP neuron pruning next.** Rank channels using weight magnitude and calibration activation
      statistics, then remove matching expansion outputs and contraction inputs (and matching
      gate/up channels for gated MLPs). Test 10%, 20%, 30% and larger cuts until quality fails.
      GPT-2's main block matrices are roughly two-thirds MLP weights: removing 30% of MLP
      channels suggests about 20% fewer block-matrix weight bytes, not a guaranteed 20% speedup
      or a claim about every architecture. Use existing dense kernels on physically smaller,
      hardware-friendly matrices; retain original weights for comparisons and recovery.

    - **Width pruning with SliceGPT and Minitron.** Keep the tokenizer initially. Reducing GPT-2's
      residual width from 768 to 256 requires compatible embeddings, normalization, residuals
      and attention/MLP projections throughout the network. Embedding columns fall to one-third;
      matrices with both dimensions reduced by three fall to one-ninth. These are size/arithmetic
      ratios, not measured speed or preserved quality. Start with 640, 512, 384, then 256.
      - [SliceGPT](https://arxiv.org/abs/2401.15024): use calibration activations and compatible
        representation transformations to concentrate useful directions, then slice rows and
        columns into smaller dense matrices. Test before recovery training; implement its
        normalization/residual handling rather than blindly deleting embedding coordinates.
      - [Minitron](https://arxiv.org/abs/2407.14679) and its
        [practical pruning/distillation report](https://arxiv.org/html/2408.11796v4): combine
        depth/width/MLP pruning with recovery distillation from the original model. Published
        examples derive roughly 4.5B models from Llama-3.1 8B. Compare calibration-only pruning
        against a fixed recovery budget on English/Strict data; account for teacher/training
        cost. This reuses pretrained weights, but recovery is not free or guaranteed.

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

    - **Speculative decoding for speed without pruning the target.** After measurement and initial
      layer/MLP experiments, use a cheap same-tokenizer draft to propose several tokens and verify
      their causal predictions in one target pass. Start with a pruned GPT-2 drafting for GPT-2;
      Qwen3-0.6B drafting for Qwen3-8B requires the 8B weights on this machine. Verify greedy
      agreement; sampling needs exact acceptance and residual correction to preserve the target
      distribution, rather than matching sampled tokens. Implement cache rollback after rejection.
      Sweep draft length and record acceptance, draft/verification time, total VRAM and net speed;
      reported 2-3x gains elsewhere are hypotheses to test here. Keep exact verification distinct
      from lossy pruning. Reference: [speculative decoding](https://arxiv.org/abs/2211.17192).
      Also try known Strict source spans as drafts and deterministic emission of fixed syntax;
      neither needs a new learned token for every block.

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
- **GPU kernels beyond NVIDIA** (found 2026-10-17): the device runtime and the tensor core kernels (`NativeRuntimeTokens.mlir`, `NativeRuntimeAttention.mlir`) call NVVM intrinsics directly. MLIR 22 lowers `gpu.subgroup_mma_*` to NVVM (wmma) but not to ROCDL ("failed to legalize gpu.subgroup_mma_load_matrix"), SPIR-V needs `spirv.entry_point_abi`. Before writing the next kernel by hand, check what MLIR already provides (`linalg` with the transform dialect's tensor core tiling, `nvgpu`/`amdgpu` mma ops), then port the existing kernels to it so AMD (ROCDL) and Vulkan (SPIR-V) get them too.
