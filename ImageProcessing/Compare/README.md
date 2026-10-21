# Blur comparison

The same 9x9 box blur of a 3840x2160 RGBA image as [BenchBlur.strict](../BenchBlur.strict), written as plain loops in other languages. Every program prints its time in ms and the center pixel (113).

| Language | Run |
|---|---|
| Strict | `strict build ImageProcessing/BenchBlur.strict`, then `ImageProcessing/BenchBlur.exe` (`STRICT_THREADS=1` for one thread, `STRICT_GPU=0` for the CPU only) |
| C | `clang -O3 -march=native blur.c` (add `-fopenmp` for the rows on all cores) |
| Rust | `rustc -C opt-level=3 -C target-cpu=native blur.rs` |
| C# | `dotnet run -c Release blur.cs` (`blur_parallel.cs` uses `Parallel.For`) |
| TypeScript | `node blur.ts` |
| Python | `python blur.py` (about 5 minutes) |
