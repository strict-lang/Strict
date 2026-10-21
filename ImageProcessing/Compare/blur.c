// Same 9x9 blur as BenchBlur.strict on Landscape.ppm, clang -O3 -march=native blur.c (-fopenmp for all cores)
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
	uint8_t* rgb = malloc(width * height * 3);
	uint8_t* data = malloc(width * height * 4);
	uint8_t* blurred = malloc(width * height * 4);
	fseek(file, header + 1, SEEK_SET);
	fread(rgb, 1, width * height * 3, file);
	fclose(file);
	for (int pixel = 0; pixel < width * height; ++pixel)
		for (int channel = 0; channel < 4; ++channel)
			data[pixel * 4 + channel] = channel < 3 ? rgb[pixel * 3 + channel] : 255;
	double started = now();
	#pragma omp parallel for
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
		{
			int red = 0, green = 0, blue = 0, alpha = 0, count = 0;
			for (int dy = -4; dy <= 4; ++dy)
				for (int dx = -4; dx <= 4; ++dx)
					if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height)
					{
						int sourceIndex = ((y + dy) * width + x + dx) * 4;
						red += data[sourceIndex];
						green += data[sourceIndex + 1];
						blue += data[sourceIndex + 2];
						alpha += data[sourceIndex + 3];
						++count;
					}
			int targetIndex = (y * width + x) * 4;
			blurred[targetIndex] = red / count;
			blurred[targetIndex + 1] = green / count;
			blurred[targetIndex + 2] = blue / count;
			blurred[targetIndex + 3] = alpha / count;
		}
	printf("C: blurred %dx%d in %.1f ms\n", width, height, (now() - started) * 1000);
	file = fopen("Landscape.c.ppm", "wb");
	fprintf(file, "P6\n%d %d\n255\n", width, height);
	for (int pixel = 0; pixel < width * height; ++pixel)
		fwrite(blurred + pixel * 4, 1, 3, file);
	fclose(file);
	return 0;
}
