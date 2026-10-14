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

/* Reads count float32 values starting start bytes into the data after the header, as doubles. */
STRICT_EXPORT double TensorFile_Floats(double self, double path, double start, double count)
{
	static strict_numbers* numbers = 0;
	uint64_t length = 0;
	uint64_t total = (uint64_t)count;
	FILE* file = Opened(strict_text(path), &length);
	if (!file)
		return strict_fail("TensorFileNotReadable");
	float* floats = malloc(total * sizeof(float) + 1);
	strict_seek(file, (int64_t)(8 + length + (uint64_t)start), SEEK_SET);
	size_t read = fread(floats, sizeof(float), total, file);
	fclose(file);
	if (read != total)
	{
		free(floats);
		return strict_fail("TensorFileDataTruncated");
	}
	free(numbers);
	numbers = malloc(sizeof(strict_numbers) + total * sizeof(double));
	numbers->count = (double)total;
	for (uint64_t index = 0; index < total; index++)
		numbers->items[index] = floats[index];
	free(floats);
	return strict_numbers_result(numbers);
}
