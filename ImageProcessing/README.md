# Strict.ImageProcessing
Images as lists of colors (`Image`, `Color`, `ColorValue`). `ImageLoader` reads and `ImageSaver` writes them through `ImageCodec`, native code around the public domain [stb_image](https://github.com/nothings/stb) in `native/vendor`, so PNG and JPG work the same on every platform.

`ImageCodec.SaveJpeg(path, pixels, width, quality)` writes JPG from flat RGB Numbers in [0,1], clamped and rounded (in the browser the file is offered as a download).

Start with `ProcessImage.strict`: `strict build ImageProcessing/ProcessImage.strict`, then `ImageProcessing/ProcessImage.exe ImageProcessing/test_image.jpg` decodes the 400x300 progressive JPG, brightens every pixel (`AdjustBrightness`) and saves `test_image_output.png`.

Every pixel is computed on its own, so the loops over pixels, rows and 8x8 blocks run in parallel on all CPU cores, with SIMD or on the GPU. `BenchBrightness.strict` runs the brightness loop alone on a blank image.

`BenchBlur.strict` is a 9x9 box blur of the 4096x2048 landscape written plainly (`Neighbors(bitmap, index).Average`, no mutables), compared with the same blur in other languages in `Compare/`, all giving the same bytes.
