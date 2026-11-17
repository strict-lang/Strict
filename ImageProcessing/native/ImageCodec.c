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

/* Rows of an image that become a JPEG or PNG strip of their own on one thread, Joined puts them into one file. */
typedef struct
{
	double pixels;
	unsigned char* bytes;
	unsigned char* data;
	size_t first, size, room, rawSize;
	int width, rows, quality, written, isPng, isFirst, isLast;
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

static void ClearFinalDeflate(unsigned char* zlib, int zlen)
{
	if ((zlib[2] & 6) == 2)
	{
		zlib[2] &= (unsigned char)~1;
		return;
	}
	int pos = 2;
	while (pos + 5 <= zlen - 4)
	{
		int len = zlib[pos + 1] | (zlib[pos + 2] << 8);
		if (zlib[pos] & 1)
			zlib[pos] &= (unsigned char)~1;
		pos += 5 + len;
	}
}

static void EncodedPng(Strip* strip)
{
	int width = strip->width, rows = strip->rows, stride = width * 4;
	unsigned char* stripBytes = strip->bytes + strip->first;
	Converted(strip->pixels, strip->first, (size_t)rows * stride, strip->bytes);
	size_t rawSize = (size_t)rows * (stride + 1);
	unsigned char* filt = malloc(rawSize);
	signed char* line = malloc(stride);
	if (!filt || !line)
	{
		free(filt);
		free(line);
		return;
	}
	for (int row = 0; row < rows; row++)
	{
		int bestFilter = 0, bestValue = 0x7fffffff, maxFilter = !strip->isFirst && row == 0 ? 2 : 5, filter = 0;
		for (; filter < maxFilter; filter++)
		{
			stbiw__encode_png_line(stripBytes, stride, width, rows, row, 4, filter, line);
			int est = 0;
			for (int i = 0; i < stride; i++)
				est += abs((signed char)line[i]);
			if (est < bestValue)
			{
				bestValue = est;
				bestFilter = filter;
			}
		}
		if (filter != bestFilter)
			stbiw__encode_png_line(stripBytes, stride, width, rows, row, 4, bestFilter, line);
		filt[(size_t)row * (stride + 1)] = (unsigned char)bestFilter;
		memcpy(filt + (size_t)row * (stride + 1) + 1, line, stride);
	}
	free(line);
	int zlen = 0;
	unsigned char* zlib = stbi_zlib_compress(filt, (int)rawSize, &zlen, stbi_write_png_compression_level);
	free(filt);
	if (!zlib || zlen < 6)
	{
		free(zlib);
		return;
	}
	if (!strip->isLast)
		ClearFinalDeflate(zlib, zlen);
	strip->data = zlib;
	strip->size = (size_t)zlen;
	strip->rawSize = rawSize;
	strip->written = 1;
}

static void Encoded(Strip* strip)
{
	if (strip->isPng)
	{
		EncodedPng(strip);
		return;
	}
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
	int rows = height < 24 ? 32 : (height / 24 + 31) / 32 * 32;
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
	strict_folders(strict_text(path));
	int written = Striped(strict_text(path), pixels, (int)width, height, (int)quality, bytes);
	free(bytes);
	return written ? 0 : strict_fail("ImageNotWritable");
}

static void WriteBe32(unsigned char* out, unsigned int value)
{
	out[0] = (unsigned char)(value >> 24);
	out[1] = (unsigned char)(value >> 16);
	out[2] = (unsigned char)(value >> 8);
	out[3] = (unsigned char)value;
}

static unsigned int CombinedAdler(unsigned int adler1, unsigned int adler2, size_t len2)
{
	const unsigned int base = 65521;
	unsigned int rem = (unsigned int)(len2 % base);
	unsigned int s1 = adler1 & 0xffff;
	unsigned int s2 = (rem * s1) % base;
	s1 += (adler2 & 0xffff) + base - 1;
	s2 += ((adler2 >> 16) & 0xffff) + ((adler1 >> 16) & 0xffff) + base - rem;
	while (s1 >= base)
		s1 -= base;
	while (s2 >= base)
		s2 -= base;
	return (s2 << 16) | s1;
}

static int JoinedPng(FILE* file, const Strip* strips, int count, int width, int height)
{
	static const unsigned char sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
	static const unsigned char iend[12] = { 0, 0, 0, 0, 'I', 'E', 'N', 'D', 0xAE, 0x42, 0x60, 0x82 };
	unsigned char ihdr[25] = { 0, 0, 0, 13, 'I', 'H', 'D', 'R' };
	WriteBe32(ihdr + 8, (unsigned int)width);
	WriteBe32(ihdr + 12, (unsigned int)height);
	ihdr[16] = 8;
	ihdr[17] = 6;
	WriteBe32(ihdr + 21, stbiw__crc32(ihdr + 4, 17));
	size_t zlen = 6;
	for (int index = 0; index < count; index++)
		zlen += strips[index].size - 6;
	unsigned char* idat = malloc(8 + zlen + 4);
	if (!idat)
		return 0;
	WriteBe32(idat, (unsigned int)zlen);
	memcpy(idat + 4, "IDAT", 4);
	idat[8] = strips[0].data[0];
	idat[9] = strips[0].data[1];
	size_t at = 10;
	unsigned int adler = 1;
	for (int index = 0; index < count; index++)
	{
		const Strip* strip = strips + index;
		size_t body = strip->size - 6;
		memcpy(idat + at, strip->data + 2, body);
		at += body;
		const unsigned char* tail = strip->data + strip->size - 4;
		unsigned int part = ((unsigned int)tail[0] << 24) | ((unsigned int)tail[1] << 16) |
			((unsigned int)tail[2] << 8) | tail[3];
		adler = index == 0 ? part : CombinedAdler(adler, part, strip->rawSize);
	}
	WriteBe32(idat + at, adler);
	WriteBe32(idat + 8 + zlen, stbiw__crc32(idat + 4, (int)(4 + zlen)));
	int written = fwrite(sig, 1, 8, file) == 8 && fwrite(ihdr, 1, 25, file) == 25 &&
		fwrite(idat, 1, 12 + zlen, file) == 12 + zlen && fwrite(iend, 1, 12, file) == 12;
	free(idat);
	return written;
}

static int StripedPng(const char* path, double pixels, int width, int height, unsigned char* bytes)
{
	int rows = height < 64 ? height : (height + 23) / 24;
	if (rows < 8)
		rows = 8;
	int count = (height + rows - 1) / rows;
	Strip strips[STRIPS] = { 0 };
	if (count < 2 || count > STRIPS)
	{
		Converted(pixels, 0, strict_count(pixels), bytes);
		return stbi_write_png(path, width, height, 4, bytes, width * 4);
	}
	for (int index = 0; index < count; index++)
	{
		int left = height - index * rows;
		strips[index] = (Strip){ pixels, bytes, 0, (size_t)index * rows * width * 4, 0, 0, 0,
			width, left < rows ? left : rows, 0, 0, 1, index == 0, index == count - 1 };
	}
	EncodedAll(strips, count);
	int written = 1;
	for (int index = 0; index < count; index++)
		written = written && strips[index].written && strips[index].data;
	FILE* file = written ? stbiw__fopen(path, "wb") : 0;
	written = file && JoinedPng(file, strips, count, width, height);
	if (file)
		written = !fclose(file) && written;
	for (int index = 0; index < count; index++)
		STBIW_FREE(strips[index].data);
	return written;
}

STRICT_EXPORT double ImageCodec_SavePng(double self, double path, double pixels, double width)
{
	int height = 0;
	const char* failure = 0;
	unsigned char* bytes = Room(pixels, width, 4, &height, &failure);
	if (!bytes)
		return strict_fail(failure);
	strict_folders(strict_text(path));
	int written = StripedPng(strict_text(path), pixels, (int)width, height, bytes);
	free(bytes);
	return written ? 0 : strict_fail("ImageNotWritable");
}
