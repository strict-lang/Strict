# Strict.ImageProcessing
Images as lists of colors (`Image`, `Color`, `ColorValue`), reading and writing PNG (`Png` with `Compression/Inflate` and `Deflate`) and reading JPG (`Jpeg`, baseline and progressive, with `JpegDecoder`, `JpegScan` and `Idct`), all written in Strict and running on the Strict VM and natively. `ImageLoader` and `ImageSaver` pick the format from the file extension.

Start with `ProcessImage.strict`: `strict build ImageProcessing/ProcessImage.strict`, then `ImageProcessing/ProcessImage.exe ImageProcessing/test_image.jpg` decodes the 400x300 progressive JPG, brightens every pixel (`AdjustBrightness`) and saves `test_image_output.png` in 0.18 s natively (27 s on the Strict VM).

Every pixel is computed on its own, which makes image processing the main use case for the for loop optimizations: the loops over pixels, rows and 8x8 blocks are where parallel `for` on all CPU cores, SIMD and the GPU should give the biggest speedups. `BenchBrightness.strict` runs the brightness loop alone on a blank image to measure that.

`BenchBlur.strict` is a 9x9 box blur of the 4096x2048 landscape written plainly (`Neighbors(bitmap, index).Average`, no mutables), compared with the same blur in `Compare/` (C, Rust, C#, TypeScript, Python, NumPy, all giving the same bytes). Measured 2026-10-21 on an i9-13900KF with an RTX 3080 (best of 3): Strict 251 ms on one thread, 26 ms on 24 threads (`STRICT_GPU=0`), 7.5 ms on the GPU (warm, median 9.1); C 282 and 53 ms with OpenMP, Rust 476 and 87, C# 856 and 122, NumPy 1678, TypeScript 1892.
