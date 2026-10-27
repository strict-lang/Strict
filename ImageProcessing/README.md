# Strict.ImageProcessing
Images as lists of colors (`Image`, `Color`, `ColorValue`), reading and writing PNG (`Png` with `Compression/Inflate` and `Deflate`) and reading JPG (`Jpeg`, baseline and progressive, with `JpegDecoder`, `JpegScan` and `Idct`), all written in Strict and running on the Strict VM and natively. `ImageLoader` picks the format from the file extension; `ImageSaver` writes PNG.

`ImageCodec.SaveJpeg(path, pixels, width, quality)` writes JPG through Windows' GDI+ codec: flat RGB Numbers in [0,1] (clamped and rounded to bytes), a positive width and quality 0-100, the height follows from the pixel count. Invalid input and write failures return an Error, other platforms `JpegRequiresWindows`. The check reads the result with Pillow:

```powershell
clang -shared -O2 -DSTRICT_LIBRARY -ICompiler ImageProcessing/native/ImageCodec.c -lgdiplus -o ImageProcessing/native/ImageProcessing.dll
python ImageProcessing/native/test_jpeg.py ImageProcessing/native/ImageProcessing.dll
```

Start with `ProcessImage.strict`: `strict build ImageProcessing/ProcessImage.strict`, then `ImageProcessing/ProcessImage.exe ImageProcessing/test_image.jpg` decodes the 400x300 progressive JPG, brightens every pixel (`AdjustBrightness`) and saves `test_image_output.png`.

Every pixel is computed on its own, so the loops over pixels, rows and 8x8 blocks run in parallel on all CPU cores, with SIMD or on the GPU. `BenchBrightness.strict` runs the brightness loop alone on a blank image.

`BenchBlur.strict` is a 9x9 box blur of the 4096x2048 landscape written plainly (`Neighbors(bitmap, index).Average`, no mutables), compared with the same blur in other languages in `Compare/`, all giving the same bytes.
