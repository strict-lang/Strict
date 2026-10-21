// Same blur as BenchBlur.strict: RGBA image of 4 bytes per pixel, box blur of (2 * radius + 1)^2 pixels, edges kept
// usage: blur width height radius passes opaque (1: alpha is always 255 and kept, only RGB is blurred)
// clang -O3 -march=native blur.c, with -fopenmp the rows run on all threads (OMP_NUM_THREADS limits them)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
static double now(void) { struct timespec time; timespec_get(&time, TIME_UTC); return time.tv_sec + time.tv_nsec * 1e-9; }
int main(int count, char** arguments)
{
	int width = count > 1 ? atoi(arguments[1]) : 3840, height = count > 2 ? atoi(arguments[2]) : 2160;
	int radius = count > 3 ? atoi(arguments[3]) : 4, passes = count > 4 ? atoi(arguments[4]) : 1;
	int opaque = count > 5 ? atoi(arguments[5]) : 0;
	long length = (long)width * height * 4, stride = (long)width * 4;
	uint8_t* pixels = malloc(length);
	uint8_t* blurred = malloc(length);
	for (long index = 0; index < length; index++) pixels[index] = opaque && index % 4 == 3 ? 255 : (uint8_t)(index * 7919 % 256);
	int area = (2 * radius + 1) * (2 * radius + 1);
	double started = now();
	for (int pass = 0; pass < passes; pass++)
	{
		#pragma omp parallel for schedule(dynamic, 16)
		for (long y = 0; y < height; y++)
			for (long x = 0; x < width; x++)
				for (long channel = 0; channel < 4; channel++)
				{
					long index = y * stride + x * 4 + channel;
					if (x < radius || x >= width - radius || y < radius || y >= height - radius || (opaque && channel == 3))
						blurred[index] = pixels[index];
					else
					{
						int sum = 0;
						for (int dy = -radius; dy <= radius; dy++)
							for (int dx = -radius; dx <= radius; dx++)
								sum += pixels[index + dy * stride + dx * 4];
						blurred[index] = (uint8_t)(sum / area);
					}
				}
		uint8_t* swap = pixels; pixels = blurred; blurred = swap;
	}
	printf("C: blurred %dx%d with %dx%d %d times in %.1f ms, pixel %d\n", width, height, 2 * radius + 1, 2 * radius + 1, passes,
		(now() - started) * 1000, pixels[length / 2 + stride / 2]);
	return 0;
}
