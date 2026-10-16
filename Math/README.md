# Strict.Math
Datatypes to make it easier to work with mathematics, algebra, equations, 2D and 3D trigonometry.
Vector2 direction constants use screen coordinates (Y down). Positive rotation maps
(1, 0) to (0, 1); games using Y up must convert at the screen boundary.

PseudoRandom ports DeltaEngine's multiply-with-carry generator. Seeded requires a
nonzero integer representable exactly by Number. Keep the state returned by Next,
then read Uint or Get(minimum, maximum). Get does not advance state; Floor converts
a sample to an integer for nonnegative ranges. Zero seeds are rejected instead of
reading wall-clock time, so simulation cannot accidentally become nondeterministic.

Vector3, Matrix, Quaternion and Ray cover 3D rendering and picking. Matrix stores 16 numbers column major like OpenGL (translation in values 12 to 14), * multiplies left to right like the C# DeltaEngine (Translation * Rotation * Scaling scales first), Inverse handles rotation, scale and translation only (no projection). Quaternion.Rotation gives its Matrix, Lerp takes the shorter way and normalizes. Ray.Ground intersects the ray with a horizontal plane (Z up).
