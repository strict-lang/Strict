#include "strict.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <pthread.h>
#endif
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
#define STRIPS 64

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

/* Room for the bytes of flat pixels, 0 and a failure name when they are not whole rows of width. */
static unsigned char* Room(double pixels, double width, int channels, int* height, const char** failure)
{
	*failure = "ImageInvalidPixelCount";
	if (!isfinite(width) || width < 1 || width > INT_MAX / channels || floor(width) != width)
		return 0;
	double rows = strict_count(pixels) / (width * channels);
	if (!isfinite(rows) || rows < 1 || rows > INT_MAX || floor(rows) != rows)
		return 0;
	*height = (int)rows;
	*failure = "ImageOutOfMemory";
	return malloc(strict_count(pixels));
}

/* Numbers from 0 to 1 (clamped and rounded) as bytes. */
static void Converted(double pixels, size_t first, size_t count, unsigned char* bytes)
{
	for (size_t index = first; index < first + count; index++)
	{
		double value = strict_number(pixels, index);
		bytes[index] = !(value > 0) ? 0 : value >= 1 ? 255 : (unsigned char)(value * 255 + 0.5);
	}
}

/* Rows of an image that become a JPEG of their own on one thread, Joined puts them into one file. */
typedef struct
{
	double pixels;
	unsigned char* bytes;
	unsigned char* data;
	size_t first, size, room;
	int width, rows, quality, written;
} Strip;

static void Appended(void* context, void* bytes, int count)
{
	Strip* strip = context;
	if (strip->size + count > strip->room)
	{
		size_t room = (strip->size + count) * 2;
		unsigned char* grown = realloc(strip->data, room);
		if (!grown)
			return;
		strip->data = grown;
		strip->room = room;
	}
	memcpy(strip->data + strip->size, bytes, count);
	strip->size += count;
}

static void Encoded(Strip* strip)
{
	Converted(strip->pixels, strip->first, (size_t)strip->rows * strip->width * 3, strip->bytes);
	strip->written = stbi_write_jpg_to_func(Appended, strip, strip->width, strip->rows, 3, strip->bytes + strip->first,
		strip->quality);
}

#if defined(_WIN32)
static DWORD WINAPI Encoding(void* strip) { Encoded(strip); return 0; }
static void EncodedAll(Strip* strips, int count)
{
	HANDLE threads[STRIPS];
	DWORD started = 0;
	for (int index = 1; index < count; index++)
		if ((threads[started] = CreateThread(0, 0, Encoding, strips + index, 0, 0)))
			started++;
		else
			Encoded(strips + index);
	Encoded(strips);
	if (started)
		WaitForMultipleObjects(started, threads, TRUE, INFINITE);
	for (DWORD index = 0; index < started; index++)
		CloseHandle(threads[index]);
}
#elif !defined(__EMSCRIPTEN__)
static void* Encoding(void* strip) { Encoded(strip); return 0; }
static void EncodedAll(Strip* strips, int count)
{
	pthread_t threads[STRIPS];
	int started[STRIPS];
	for (int index = 1; index < count; index++)
		if (!(started[index] = !pthread_create(threads + index, 0, Encoding, strips + index)))
			Encoded(strips + index);
	Encoded(strips);
	for (int index = 1; index < count; index++)
		if (started[index])
			pthread_join(threads[index], 0);
}
#else
static void EncodedAll(Strip* strips, int count)
{
	for (int index = 0; index < count; index++)
		Encoded(strips + index);
}
#endif

/* Start of the segment with this marker in the JPEG of a strip, 0 when it has none. */
static size_t Segment(const Strip* strip, int marker)
{
	size_t at = 2;
	while (at + 4 <= strip->size && strip->data[at] == 0xFF && strip->data[at + 1] != marker)
		at += 2 + ((size_t)strip->data[at + 2] << 8 | strip->data[at + 3]);
	return at + 4 <= strip->size && strip->data[at] == 0xFF ? at : 0;
}

/* The JPEG of the first strip with the full height and a restart interval of one strip, then the scan of every other
   strip (after its scan header) behind a restart marker, each without its end marker. */
static int Joined(FILE* file, Strip* strips, int count, int height, int interval)
{
	static const unsigned char end[2] = { 0xFF, 0xD9 };
	size_t frame = Segment(strips, 0xC0), scan = Segment(strips, 0xDA);
	if (!frame || !scan)
		return 0;
	strips->data[frame + 5] = (unsigned char)(height >> 8);
	strips->data[frame + 6] = (unsigned char)height;
	unsigned char restart[6] = { 0xFF, 0xDD, 0, 4, (unsigned char)(interval >> 8), (unsigned char)interval };
	int written = fwrite(strips->data, 1, scan, file) == scan && fwrite(restart, 1, 6, file) == 6;
	for (int index = 0; index < count && written; index++)
	{
		const Strip* strip = strips + index;
		unsigned char marker[2] = { 0xFF, (unsigned char)(0xD0 + (index + 7) % 8) };
		size_t header = Segment(strip, 0xDA);
		size_t start = index ? header + 2 + ((size_t)strip->data[header + 2] << 8 | strip->data[header + 3]) : scan;
		written = header && strip->size >= start + 2 && (index == 0 || fwrite(marker, 1, 2, file) == 2) &&
			fwrite(strip->data + start, 1, strip->size - 2 - start, file) == strip->size - 2 - start;
	}
	return written && fwrite(end, 1, 2, file) == 2;
}

/* A JPEG encoded as about 24 strips of whole blocks in parallel, which decodes to the same pixels as one pass over
   all rows; below 64 rows that one pass. */
static int Striped(const char* path, double pixels, int width, int height, int quality, unsigned char* bytes)
{
	int block = quality <= 90 ? 16 : 8;
	int rows = (height / 24 + 31) / 32 * 32;
	int count = (height + rows - 1) / rows;
	int interval = (width + block - 1) / block * (rows / block);
	Strip strips[STRIPS] = { 0 };
	if (count < 2 || count > STRIPS || interval > 65535)
	{
		Converted(pixels, 0, (size_t)width * height * 3, bytes);
		return stbi_write_jpg(path, width, height, 3, bytes, quality);
	}
	for (int index = 0; index < count; index++)
	{
		int left = height - index * rows;
		strips[index] = (Strip){ pixels, bytes, 0, (size_t)index * rows * width * 3, 0, 0, width, left < rows ? left : rows,
			quality, 0 };
	}
	EncodedAll(strips, count);
	int written = 1;
	for (int index = 0; index < count; index++)
		written = written && strips[index].written && strips[index].data;
	FILE* file = written ? stbiw__fopen(path, "wb") : 0;
	written = file && Joined(file, strips, count, height, interval);
	if (file)
		written = !fclose(file) && written;
	for (int index = 0; index < count; index++)
		free(strips[index].data);
	return written;
}

STRICT_EXPORT double ImageCodec_SaveJpeg(double self, double path, double pixels, double width, double quality)
{
	if (!isfinite(quality) || quality < 0 || quality > 100 || floor(quality) != quality)
		return strict_fail("JpegInvalidQuality");
	int height = 0;
	const char* failure = 0;
	unsigned char* bytes = Room(pixels, width, 3, &height, &failure);
	if (!bytes)
		return strict_fail(failure);
	int written = Striped(strict_text(path), pixels, (int)width, height, (int)quality, bytes);
	free(bytes);
	return written ? 0 : strict_fail("ImageNotWritable");
}

STRICT_EXPORT double ImageCodec_SavePng(double self, double path, double pixels, double width)
{
	int height = 0;
	const char* failure = 0;
	unsigned char* bytes = Room(pixels, width, 4, &height, &failure);
	if (!bytes)
		return strict_fail(failure);
	Converted(pixels, 0, strict_count(pixels), bytes);
	int written = stbi_write_png(strict_text(path), (int)width, height, 4, bytes, (int)width * 4);
	free(bytes);
	return written ? 0 : strict_fail("ImageNotWritable");
}
