# Model improvements
Details for the model line of [improvements.md](improvements.md). Remove what is done, add new findings at the end of their section, bugs go into improvements.md.

## Ternary Bonsai 27B
`C:/code/Models/Bonsai`, 12.7-18 ms per token on the RTX 4090 (Prism's llama.cpp ~11); per token (ncu) the products take ~8.7 ms, the quantize kernels ~0.6. Left:
- the dot products decode every PTQ1_0 block twice, the two lanes of a block should split the codes
- one kernel for 1 to 8 tokens (like Kindi's) so `LookupDecoding` verifies cheaper (2, 4, 8 tokens 27.7, 37.3, 47.2 ms against 16.8 for one)
- CPU loops unpack 3 bit rows to doubles (`MlirWidths` decodes only 4 and 6 bit blocks per use)
- the 257 `Sylvester` launches per token still take ~1.1 ms (4.4 us each with one barrier, latency bound): rotate the inputs of a block in fewer launches
- 10 GB budget (2026-10-27, holder leaves 10.0 GB free, 64 tokens): the run right after the first build (its PTX compiled, 2 min) stalled with a raw cuMemAlloc holder (median 15 ms, mean 119, p95 744), a second one with a raw and a torch holder alive together (mean 144, p95 925); 12 later runs, raw and torch holders alternating, did not (64 tokens mean 15-20 ms, p95 31-35; 256 tokens mean 14.0-16.8, p95 17.8-21.6; without a holder median 11.8, mean 13.8). The raw holder's prefill was 11-13 ms per token in 4 of 5 64 token runs against 4.2 with torch, equal in the 256 token runs. Next: log per token times and `strict_gpu_starved` evictions of a cold run (fresh PTX) under a raw holder to see where the stalled tokens wait, before changing eviction or allocation

## Qwen3.8-Flash-Next
qwen4exp, `C:/code/Models/FlashNext`, ~280 ms per token, 325 with MTP drafts. Left:
- draft acceptance on a chat prompt with a longer answer and with 3 drafts (llama.cpp reports 89 %, here 7 and 12 of 24)
- the 3 token `Steps` costs ~1.5 plain steps
- the QSA indexer past 2051 tokens
