# Strict.Math
Datatypes to make it easier to work with mathematics, algebra, equations, 2D and 3D trigonometry.
`Vector2(x, y).Atan` is the arc tangent of y / x in degrees from +Y towards +X (`Vector2(1, 0).Atan is 90`), the arc tangent of any number is `Vector2(value, 1).Atan` (`Ratio.Atan` only takes -1 to 1). Vector2 direction constants use screen coordinates (Y down). Positive rotation maps
(1, 0) to (0, 1); games using Y up must convert at the screen boundary.

Vector3, Matrix, Quaternion and Ray cover 3D rendering and picking. Matrix stores 16 numbers column major like OpenGL (translation in values 12 to 14), * multiplies left to right like the C# DeltaEngine (Translation * Rotation * Scaling scales first), Inverse handles rotation, scale and translation only (no projection). Perspective and Orthographic build the projection matrices (OpenGL clip space, depth -1 to 1). Quaternion.Rotation gives its Matrix, Lerp takes the shorter way and normalizes. Ray.Ground intersects the ray with a horizontal plane (Z up).

DoubleDouble keeps high and low double components for finite values in fractal arithmetic. Mandelbrot provides escape counts, cardioid/bulb rejection, high-precision reference orbits, guarded perturbation and a shared finite-iteration disk enclosure. FractalZoomer in DeltaEngine uses these without duplicating arithmetic in game code.
