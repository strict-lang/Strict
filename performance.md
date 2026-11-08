# Performance
Open speed work that can be done soon, one short line each, measurements go into commit messages. GPU details: [improvements-gpu.md](improvements-gpu.md), models: [improvements-models.md](improvements-models.md). Anything large or difficult: [ideas.md](ideas.md).

## Soon
- MossWeb 480p after 720p images in the same page is 15-20 % slower than in a fresh page (fastest 5.1-5.7 instead of 4.3-4.5 s an image on a 3080): the WebGPU buffer pool probably keeps the 720p sized buffers, smaller sizes should reuse or shrink them.
