# Performance
Open speed work that can be done soon, one short line each, measurements go into commit messages. GPU details: [improvements-gpu.md](improvements-gpu.md), models: [improvements-models.md](improvements-models.md). Anything large or difficult: [ideas.md](ideas.md).

## Soon
- Testing one file on the VM spends 0.7-4.3 s before its first test (Compiler/MlirLibrary 4.3 s, MlirCalls 2.7 s), all of it again every run: TypeShape.MemberScope is rebuilt 1191 times and TypeCodegen.Generator 1239 times for Number.strict, Changes(lines).Members and .Methods per method body and call site, source discovery tokenizes every line with a `/`, ResolveCheck.Inference covers all project files. Compute shape scopes once per TypeCodegen, cache per file scopes in the server by file stamp.
- GPU preparation takes 0.5-0.6 s of every build with kernels also when nothing changed (`DeviceBuild.Built`: `PtxCache` reads the runtime PTX and hashes the kernel text with `HashCode` per character), `MlirSplit` still reads the just written module back.
- MossWeb 480p after 720p images in the same page is 15-20 % slower than in a fresh page (fastest 5.1-5.7 instead of 4.3-4.5 s an image on a 3080): the WebGPU buffer pool probably keeps the 720p sized buffers, smaller sizes should reuse or shrink them.
- Loading Moss with several LoRAs got about 2 s slower since 2026-11-04 (three LoRAs 8.6-9.1 s before, 10.1-11.1 s now): find what changed.
