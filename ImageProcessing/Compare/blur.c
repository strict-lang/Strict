// Same blur as BenchBlur.strict: 1920x1080 RGB doubles, 3x3 box blur, 10 passes, edges kept
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
enum { Width = 1920, Height = 1080, Passes = 10, Stride = Width * 3, Count = Width * Height * 3 };
static double now(void) { struct timespec time; timespec_get(&time, TIME_UTC); return time.tv_sec + time.tv_nsec * 1e-9; }
int main(void)
{
	double* pixels = malloc(Count * sizeof(double));
	double* blurred = malloc(Count * sizeof(double));
	for (long index = 0; index < Count; index++) pixels[index] = (index * 7919 % 256) / 255.0;
	double started = now();
	for (int pass = 0; pass < Passes; pass++)
	{
		for (long index = 0; index < Count; index++)
		{
			long column = index % Stride;
			if (index < Stride || index >= Count - Stride || column < 3 || column >= Stride - 3)
				blurred[index] = pixels[index];
			else
				blurred[index] = (pixels[index - Stride - 3] + pixels[index - Stride] + pixels[index - Stride + 3] +
					pixels[index - 3] + pixels[index] + pixels[index + 3] +
					pixels[index + Stride - 3] + pixels[index + Stride] + pixels[index + Stride + 3]) / 9;
		}
		double* swap = pixels; pixels = blurred; blurred = swap;
	}
	printf("C: blurred %dx%d %d times in %.1f ms, pixel %.16g\n", Width, Height, Passes, (now() - started) * 1000, pixels[3000003]);
	return 0;
}
