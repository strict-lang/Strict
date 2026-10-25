# Strict.ImageProcessing
Images as lists of colors (`Image`, `Color`, `ColorValue`), reading and writing PNG (`Png` with `Compression/Inflate` and `Deflate`) and reading JPG (`Jpeg`, baseline and progressive, with `JpegDecoder`, `JpegScan` and `Idct`), all written in Strict and running on the Strict VM and natively. `ImageLoader` picks the format from the file extension; `ImageSaver` writes PNG.

`ImageCodec.SaveJpeg(path, pixels, width, quality)` uses Windows' built-in GDI+ codec without an external dependency. Pass flat RGB Numbers in [0,1], a positive integer width and integer quality 0-100; height follows from the pixel count. Values are clamped and rounded to bytes. Invalid dimensions, quality, paths and write failures return Error. Other platforms report JpegRequiresWindows. The existing Strict JPEG reader remains unchanged. The codec check uses Pillow to read the result:

```powershell
clang -shared -O2 -DSTRICT_LIBRARY -ICompiler ImageProcessing/native/ImageCodec.c -lgdiplus -o ImageProcessing/native/ImageProcessing.dll
python ImageProcessing/native/test_jpeg.py ImageProcessing/native/ImageProcessing.dll
```

Start with `ProcessImage.strict`: `strict build ImageProcessing/ProcessImage.strict`, then `ImageProcessing/ProcessImage.exe ImageProcessing/test_image.jpg` decodes the 400x300 progressive JPG, brightens every pixel (`AdjustBrightness`) and saves `test_image_output.png` in 0.18 s natively (27 s on the Strict VM).

Every pixel is computed on its own, which makes image processing the main use case for the for loop optimizations: the loops over pixels, rows and 8x8 blocks are where parallel `for` on all CPU cores, SIMD and the GPU should give the biggest speedups. `BenchBrightness.strict` runs the brightness loop alone on a blank image to measure that.

`BenchBlur.strict` is a 9x9 box blur of the 4096x2048 landscape written plainly (`Neighbors(bitmap, index).Average`, no mutables), compared with the same blur in `Compare/` (C, Rust, Goose, C#, TypeScript, Python, NumPy, all giving the same bytes): Strict is as fast as C with clang on one thread, faster than C with OpenMP on all cores and faster again on the GPU.
