# Model improvements
Details for the model line of [improvements.md](improvements.md). Remove what is done, add new findings at the end of their section, bugs go into improvements.md.

## Ternary Bonsai 27B
`C:/code/Models/Bonsai`, 12.7-18 ms per token on the RTX 4090 (Prism's llama.cpp ~11); per token (ncu) the products take ~8.7 ms, the quantize kernels ~0.6. Left:
- the dot products decode every PTQ1_0 block twice, the two lanes of a block should split the codes
- one kernel for 1 to 8 tokens (like Kindi's) so `LookupDecoding` verifies cheaper (2, 4, 8 tokens 27.7, 37.3, 47.2 ms against 16.8 for one)
- CPU loops unpack 3 bit rows to doubles (`MlirWidths` decodes only 4 and 6 bit blocks per use)
- the 257 `Sylvester` launches per token still take ~1.1 ms (4.4 us each with one barrier, latency bound): rotate the inputs of a block in fewer launches
- 10 GB budget: a holder process does not simulate it on Windows (the budget never saw the holder, see improvements.md), capping the driver's free memory at 10 GB in `strict_gpu_free_memory` does (budget 8.5 GB, peak 10.0 GB in use, the same in cold and warm runs). Then 64 tokens decode in a median of 22-24 instead of 12-14 ms: per token 3-4 of the 12 CUDA graphs are instantiated anew (5 ms, 1 ms destroying, 9 instead of 12 updates), 3-4 cuMemAlloc/cuMemFree (2 ms), the result download waits 7 instead of 3 ms; 2 tokens per run take ~290 ms freeing and allocating ~100 buffers (one uploads 298 MB)

## Qwen3.8-Flash-Next
qwen4exp, `C:/code/Models/FlashNext`, ~280 ms per token, 325 with MTP drafts. Left:
- draft acceptance on a chat prompt with a longer answer and with 3 drafts (llama.cpp reports 89 %, here 7 and 12 of 24)
- the 3 token `Steps` costs ~1.5 plain steps
- the QSA indexer past 2051 tokens
