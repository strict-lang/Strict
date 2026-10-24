# Strict.ImageProcessing
Images as lists of colors (`Image`, `Color`, `ColorValue`), reading and writing PNG (`Png` with `Compression/Inflate` and `Deflate`) and reading JPG (`Jpeg`, baseline and progressive, with `JpegDecoder`, `JpegScan` and `Idct`), all written in Strict and running on the Strict VM and natively. `ImageLoader` picks the format from the file extension; `ImageSaver` writes PNG.

`ImageCodec.SaveJpeg(path, pixels, width, quality)` uses Windows' built-in GDI+ codec without an external dependency. Pass flat RGB Numbers in [0,1], a positive integer width and integer quality 0-100; height follows from the pixel count. Values are clamped and rounded to bytes. Invalid dimensions, quality, paths and write failures return Error. Other platforms report JpegRequiresWindows. The existing Strict JPEG reader remains unchanged. The codec check uses Pillow to read the result:

```powershell
clang -shared -O2 -DSTRICT_LIBRARY -ICompiler ImageProcessing/native/ImageCodec.c -lgdiplus -o ImageProcessing/native/ImageProcessing.dll
python ImageProcessing/native/test_jpeg.py ImageProcessing/native/ImageProcessing.dll
```

Start with `ProcessImage.strict`: `strict build ImageProcessing/ProcessImage.strict`, then `ImageProcessing/ProcessImage.exe ImageProcessing/test_image.jpg` decodes the 400x300 progressive JPG, brightens every pixel (`AdjustBrightness`) and saves `test_image_output.png` in 0.18 s natively (27 s on the Strict VM).

Every pixel is computed on its own, which makes image processing the main use case for the for loop optimizations: the loops over pixels, rows and 8x8 blocks are where parallel `for` on all CPU cores, SIMD and the GPU should give the biggest speedups. `BenchBrightness.strict` runs the brightness loop alone on a blank image to measure that.

`BenchBlur.strict` is a 9x9 box blur of the 4096x2048 landscape written plainly (`Neighbors(bitmap, index).Average`, no mutables), compared with the same blur in `Compare/` (C, Rust, Goose, C#, TypeScript, Python, NumPy, all giving the same bytes). Measured 2026-10-23 on an i9-13900KF with an RTX 3080 (best of 3, LLVM 23.1.3): Strict 190 ms on one thread (the pixel loop counts x and y along with the index, `index % Size.Width` and `index / Size.Width` in the called methods become x and y), 21 ms on 24 threads (`STRICT_GPU=0`), 6.6 ms on the GPU (Run blurs twice and times the second call, the first one opens the GPU); C with clang 23 193 and 58 ms with OpenMP (clang 22 on 2026-10-21: 282 and 53), Rust 488 and 75, Goose 584 (518 with `-O3 -march=native`, single threaded since workers share no memory), C# 851, NumPy 1648, TypeScript 1960.
