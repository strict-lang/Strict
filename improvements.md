# Improvements
Bugs to fix, most harmful first, one short line each. Remove a line once it is done, add what is found. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md). Anything large or difficult, features and far fetched things: [ideas.md](ideas.md).

- A method named `Length` in a type with a list member resolves to something else inside its own methods (Math/Perturbation with `has orbit Float32s` and `Length Number = orbit.Length / 2`: every pixel escaped after one step, calls from outside worked); renamed to Points it works.
- `.Sum` of the Float32s from Perturbation.Strip gives 0 with non-zero elements, on the GPU and with STRICT_GPU=0; a `for` loop adding them gives the right total.
- `strict test Math/Mandelbrot.strict` fails with 3 MutationSurvived (Escape line 13, Iterated line 34 and one more).
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
