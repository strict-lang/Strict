#include "strict.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STB_IMAGE_STATIC
#define STBI_WINDOWS_UTF8
#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"
#define STB_IMAGE_WRITE_STATIC
#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

/* Width or height of the image at path without decoding it, 0 when it can't be read. */
static int Measured(double path, int* height)
{
	int width = 0, channels = 0;
	return stbi_info(strict_text(path), &width, height, &channels) ? width : 0;
}

STRICT_EXPORT double ImageCodec_Width(double self, double path)
{
	int height = 0, width = Measured(path, &height);
	return width ? (double)width : strict_fail("ImageNotReadable");
}

STRICT_EXPORT double ImageCodec_Height(double self, double path)
{
	int height = 0, width = Measured(path, &height);
	return width ? (double)height : strict_fail("ImageNotReadable");
}

/* The pixels from the top left as numbers from 0 to 255: 1 is gray, 3 RGB, 4 RGBA. */
STRICT_EXPORT double ImageCodec_Pixels(double self, double path, double channels)
{
	static _Thread_local strict_packed* kept = 0;
	if (channels != 1 && channels != 3 && channels != 4)
		return strict_fail("ImageChannelsMustBe1Or3Or4");
	int width = 0, height = 0, stored = 0;
	unsigned char* pixels = stbi_load(strict_text(path), &width, &height, &stored, (int)channels);
	if (!pixels)
		return strict_fail("ImageNotReadable");
	uint64_t count = (uint64_t)width * (uint64_t)height * (uint64_t)channels;
	free(kept);
	kept = malloc(sizeof(strict_packed) + count * sizeof(float));
	if (!kept)
	{
		stbi_image_free(pixels);
		return strict_fail("ImageOutOfMemory");
	}
	kept->count = (double)count;
	kept->bits = 32;
	float* numbers = (float*)kept->data;
	for (uint64_t index = 0; index < count; index++)
		numbers[index] = pixels[index];
	stbi_image_free(pixels);
	return strict_packed_result(kept);
}

/* Flat numbers from 0 to 1 (clamped and rounded) as bytes, 0 and a failure name when they don't fit width. */
static unsigned char* Bytes(double pixels, double width, int channels, int* height, const char** failure)
{
	*failure = "ImageInvalidPixelCount";
	if (!isfinite(width) || width < 1 || width > INT_MAX / channels || floor(width) != width)
		return 0;
	double rows = strict_count(pixels) / (width * channels);
	if (!isfinite(rows) || rows < 1 || rows > INT_MAX || floor(rows) != rows)
		return 0;
	*height = (int)rows;
	size_t count = strict_count(pixels);
	unsigned char* bytes = malloc(count);
	if (!bytes)
	{
		*failure = "ImageOutOfMemory";
		return 0;
	}
	for (size_t index = 0; index < count; index++)
	{
		double value = strict_number(pixels, index);
		bytes[index] = !(value > 0) ? 0 : value >= 1 ? 255 : (unsigned char)(value * 255 + 0.5);
	}
	return bytes;
}

STRICT_EXPORT double ImageCodec_SaveJpeg(double self, double path, double pixels, double width, double quality)
{
	if (!isfinite(quality) || quality < 0 || quality > 100 || floor(quality) != quality)
		return strict_fail("JpegInvalidQuality");
	int height = 0;
	const char* failure = 0;
	unsigned char* bytes = Bytes(pixels, width, 3, &height, &failure);
	if (!bytes)
		return strict_fail(failure);
	int written = stbi_write_jpg(strict_text(path), (int)width, height, 3, bytes, (int)quality);
	free(bytes);
	return written ? 0 : strict_fail("ImageNotWritable");
}

STRICT_EXPORT double ImageCodec_SavePng(double self, double path, double pixels, double width)
{
	int height = 0;
	const char* failure = 0;
	unsigned char* bytes = Bytes(pixels, width, 4, &height, &failure);
	if (!bytes)
		return strict_fail(failure);
	int written = stbi_write_png(strict_text(path), (int)width, height, 4, bytes, (int)width * 4);
	free(bytes);
	return written ? 0 : strict_fail("ImageNotWritable");
}
