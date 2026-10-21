// Same blur as BenchBlur.strict on Landscape.ppm: box blur of (2 * radius + 1)^2 pixels per RGB channel, edges kept
// clang -O3 -march=native blur.c, with -fopenmp the rows run on all threads (OMP_NUM_THREADS limits them)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
static double now(void) { struct timespec time; timespec_get(&time, TIME_UTC); return time.tv_sec + time.tv_nsec * 1e-9; }
int main(void)
{
	FILE* file = fopen("Landscape.ppm", "rb");
	int width, height, maximum, header;
	if (!file || fscanf(file, "P6 %d %d %d%n", &width, &height, &maximum, &header) != 3) return 1;
	long length = (long)width * height * 3, stride = (long)width * 3;
	uint8_t* pixels = malloc(length);
	uint8_t* blurred = malloc(length);
	fseek(file, header + 1, SEEK_SET);
	fread(pixels, 1, length, file);
	fclose(file);
	int radius = 4, area = (2 * radius + 1) * (2 * radius + 1);
	double started = now();
	#pragma omp parallel for schedule(dynamic, 16)
	for (long y = 0; y < height; y++)
		for (long x = 0; x < width; x++)
			for (long channel = 0; channel < 3; channel++)
			{
				long index = y * stride + x * 3 + channel;
				if (x < radius || x >= width - radius || y < radius || y >= height - radius)
					blurred[index] = pixels[index];
				else
				{
					int sum = 0;
					for (int dy = -radius; dy <= radius; dy++)
						for (int dx = -radius; dx <= radius; dx++)
							sum += pixels[index + dy * stride + dx * 3];
					blurred[index] = (uint8_t)(sum / area);
				}
			}
	printf("C: blurred %dx%d in %.1f ms\n", width, height, (now() - started) * 1000);
	file = fopen("Landscape.c.ppm", "wb");
	fprintf(file, "P6\n%d %d\n255\n", width, height);
	fwrite(blurred, 1, length, file);
	fclose(file);
	return 0;
}
