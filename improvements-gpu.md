# GPU improvements
Details for the GPU line of [improvements.md](improvements.md), one short line each, measurements go into commit messages. Diffusion models (Moss, H3) live in `../StrictComfy/`. Remove what is done, bugs go into improvements.md.

- no native test reaches the locked path of `strict_packed_rows` and `strict_rows` (matrices of 4 MB and more): a test with a big matrix
