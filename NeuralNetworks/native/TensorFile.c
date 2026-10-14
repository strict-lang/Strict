#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "strict.h"
#ifdef _WIN32
#define strict_seek _fseeki64
#else
#define strict_seek fseeko
#endif

/* Opens a .safetensors file and reads the length of its JSON header (8 bytes little endian). */
static FILE* Opened(const char* path, uint64_t* headerLength)
{
	FILE* file = fopen(path, "rb");
	unsigned char size[8];
	if (!file)
		return 0;
	if (fread(size, 1, 8, file) != 8)
	{
		fclose(file);
		return 0;
	}
	*headerLength = 0;
	for (int index = 7; index >= 0; index--)
		*headerLength = *headerLength * 256 + size[index];
	return file;
}

STRICT_EXPORT double TensorFile_Header(double self, double path)
{
	static char* header = 0;
	uint64_t length = 0;
	FILE* file = Opened(strict_text(path), &length);
	if (!file || length > 100000000)
	{
		if (file)
			fclose(file);
		return strict_fail("TensorFileNotReadable");
	}
	free(header);
	header = malloc(length + 1);
	size_t read = fread(header, 1, length, file);
	fclose(file);
	if (read != length)
		return strict_fail("TensorFileHeaderTruncated");
	header[length] = 0;
	return strict_text_result(header);
}

/* Reads count float32 values starting start bytes into the data after the header, 0 and the failure if it can't. */
static float* Read(double path, double start, uint64_t count, const char** failure)
{
	uint64_t length = 0;
	FILE* file = Opened(strict_text(path), &length);
	if (!file)
	{
		*failure = "TensorFileNotReadable";
		return 0;
	}
	float* floats = malloc(count * sizeof(float) + 1);
	strict_seek(file, (int64_t)(8 + length + (uint64_t)start), SEEK_SET);
	size_t read = fread(floats, sizeof(float), count, file);
	fclose(file);
	if (read == count)
		return floats;
	free(floats);
	*failure = "TensorFileDataTruncated";
	return 0;
}

/* Reads count float32 values starting start bytes into the data after the header, as doubles. */
STRICT_EXPORT double TensorFile_Floats(double self, double path, double start, double count)
{
	static strict_numbers* numbers = 0;
	const char* failure = 0;
	uint64_t total = (uint64_t)count;
	float* floats = Read(path, start, total, &failure);
	if (!floats)
		return strict_fail(failure);
	free(numbers);
	numbers = malloc(sizeof(strict_numbers) + total * sizeof(double));
	numbers->count = (double)total;
	for (uint64_t index = 0; index < total; index++)
		numbers->items[index] = floats[index];
	free(floats);
	return strict_numbers_result(numbers);
}

/* The rows (or the columns when transposed) of the float32 matrix of shape (rows, columns) at start, each as numbers with
   bits 32 (exact) or 16 (rounded to halves). */
static double Matrix(double path, double start, double shape, double bits, int transposed)
{
	static double* rows = 0;
	static unsigned char* storage = 0;
	const strict_numbers* dimensions = strict_numbers_of(shape);
	if (dimensions->count != 2 || (bits != 16 && bits != 32))
		return strict_fail("OnlyMatricesOf16Or32BitsAreSupported");
	uint64_t height = (uint64_t)dimensions->items[0], width = (uint64_t)dimensions->items[1];
	const char* failure = 0;
	float* floats = Read(path, start, height * width, &failure);
	if (!floats)
		return strict_fail(failure);
	uint64_t outer = transposed ? width : height, inner = transposed ? height : width;
	uint64_t size = (sizeof(strict_packed) + inner * ((uint64_t)bits / 8) + 7) / 8 * 8;
	free(rows);
	free(storage);
	rows = malloc((1 + outer) * sizeof(double));
	storage = malloc(outer * size + 1);
	rows[0] = (double)outer;
	for (uint64_t row = 0; row < outer; row++)
	{
		strict_packed* packed = (strict_packed*)(storage + row * size);
		packed->count = (double)inner;
		packed->bits = bits;
		for (uint64_t column = 0; column < inner; column++)
		{
			float value = transposed ? floats[column * width + row] : floats[row * width + column];
			if (bits == 16)
				((uint16_t*)packed->data)[column] = strict_half(value);
			else
				((float*)packed->data)[column] = value;
		}
		rows[1 + row] = strict_packed_result(packed);
	}
	free(floats);
	return strict_numbers_result((strict_numbers*)rows);
}

STRICT_EXPORT double TensorFile_Rows(double self, double path, double start, double shape, double bits)
{
	return Matrix(path, start, shape, bits, 0);
}

STRICT_EXPORT double TensorFile_Columns(double self, double path, double start, double shape, double bits)
{
	return Matrix(path, start, shape, bits, 1);
}
