# Blur comparison

Every program loads `Landscape.ppm` (4096x2048 RGB, made from the Armies of Steel landscape in the DeltaEngine repository), blurs it with a 9x9 box per color channel (edges kept), prints the time of the blur and writes `Landscape.<language>.ppm`. All outputs are byte for byte the same.

Create the input once (Python with Pillow): `python -c "from PIL import Image; Image.open('../../../DeltaEngine/Games/ArmiesOfSteel/Content/LandscapeAll.jpg').convert('RGB').save('Landscape.ppm')"`

| Language | Run in this folder |
|---|---|
| Strict | `strict build ImageProcessing/BenchBlur.strict` in the repository root, then `ImageProcessing/BenchBlur.exe` there (`STRICT_THREADS=1` for one thread, `STRICT_GPU=0` for the CPU only) |
| C | `clang -O3 -march=native blur.c` (add `-fopenmp` for the rows on all cores) |
| Rust | `rustc -C opt-level=3 -C target-cpu=native blur.rs` (`blur_parallel.rs` splits the rows over all cores with scoped threads) |
| Goose | `goose build blur.goose --cc clang -o blur_goose.exe` ([goose](https://github.com/aardappel/goose) built with CMake) |
| C# | `dotnet run -c Release blur.cs` (`blur_parallel.cs` uses `Parallel.For`) |
| TypeScript | `node blur.ts` |
| Python | `python blur.py` (a few minutes), `python blur_numpy.py` adds each of the 81 offsets for all pixels at once with NumPy |

Every version blurs the raw RGBA bytes (a PPM has no alpha, it is loaded as 255) with the same loop and no hand optimizations. Strict averages the Colors of the neighbors inside the image (`Neighbors(bitmap, index).Average`), the compiler turns that into one loop with a sum per member, keeps a Color as one whole number and picks the threads.
