#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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

/* A tensor as its header entry describes it: "dtype":"F16","shape":[2,3],"data_offsets":[24,36]. */
typedef struct { uint64_t bytes, isBrain, isSigned, rank, height, width, start; } Tensor;

/* Reads the numbers of a list like [2,3] at text into numbers (at most limit), their count or -1 when it is malformed. */
static int Listed(const char* text, uint64_t* numbers, int limit)
{
	int count = 0;
	char* next = (char*)text;
	while (*next != ']')
	{
		if (count == limit || *next < '0' || *next > '9')
			return -1;
		numbers[count++] = strtoull(next, &next, 10);
		if (*next == ',')
			next++;
		else if (*next != ']')
			return -1;
	}
	return count;
}

/* Fills tensor from entry, 0 or the name of what is wrong with it. */
static const char* Described(const char* entry, Tensor* tensor)
{
	const char* type = strstr(entry, "\"dtype\":\"");
	const char* shape = strstr(entry, "\"shape\":[");
	const char* offsets = strstr(entry, "\"data_offsets\":[");
	if (!type || !shape || !offsets)
		return "TensorNotFound";
	type += 9;
	tensor->isBrain = strncmp(type, "BF16\"", 5) == 0;
	tensor->isSigned = strncmp(type, "I8\"", 3) == 0;
	tensor->bytes = strncmp(type, "F32\"", 4) == 0 ? 4 : strncmp(type, "F16\"", 4) == 0 || tensor->isBrain ? 2 :
		tensor->isSigned || strncmp(type, "U8\"", 3) == 0 ? 1 : 0;
	if (!tensor->bytes)
		return "OnlyF32F16BF16I8AndU8TensorsAreSupported";
	uint64_t dimensions[4] = { 1, 1, 1, 1 }, range[2];
	int rank = Listed(shape + 9, dimensions, 4);
	if (rank < 0 || Listed(offsets + 16, range, 2) != 2)
		return "OnlyTensorsUpToFourDimensionsAreSupported";
	tensor->rank = (uint64_t)rank;
	tensor->width = rank > 0 ? dimensions[rank - 1] : 1;
	tensor->height = 1;
	for (int dimension = 0; dimension + 1 < rank; dimension++)
		tensor->height *= dimensions[dimension];
	tensor->start = range[0];
	if (tensor->height > (1ull << 30) || tensor->width > (1ull << 30))
		return "TensorTooLarge";
	if (range[1] < range[0] || range[1] - range[0] != tensor->height * tensor->width * tensor->bytes)
		return "TensorSizeMismatch";
	return 0;
}

/* Reads the numbers of a tensor as 32 bit floats, 0 and the failure if it can't. */
static float* Read(double path, const Tensor* tensor, const char** failure)
{
	uint64_t length = 0, count = tensor->height * tensor->width;
	FILE* file = Opened(strict_text(path), &length);
	if (!file)
	{
		*failure = "TensorFileNotReadable";
		return 0;
	}
	float* floats = malloc(count * sizeof(float) + 1);
	if (!floats)
	{
		fclose(file);
		*failure = "TensorTooLarge";
		return 0;
	}
	strict_seek(file, (int64_t)(8 + length + tensor->start), SEEK_SET);
	size_t read = fread(floats, (size_t)tensor->bytes, count, file);
	fclose(file);
	if (read != count)
	{
		free(floats);
		*failure = "TensorFileDataTruncated";
		return 0;
	}
	/* 8 and 16 bit numbers fill the start, widened from the last one down so none is overwritten before it is read */
	const uint16_t* halves = (const uint16_t*)floats;
	const unsigned char* bytes = (const unsigned char*)floats;
	for (uint64_t index = count; tensor->bytes == 2 && index-- > 0;)
		floats[index] = tensor->isBrain ? strict_brain_float(halves[index]) : strict_half_float(halves[index]);
	for (uint64_t index = count; tensor->bytes == 1 && index-- > 0;)
		floats[index] = tensor->isSigned ? (float)(int8_t)bytes[index] : (float)bytes[index];
	return floats;
}

/* The numbers of the tensor that entry describes: 32 bit floats for float tensors (exact, the GPU keeps them as they
   are), doubles for integers. */
STRICT_EXPORT double TensorFile_Floats(double self, double path, double entry)
{
	static unsigned char* storage = 0;
	const char* failure = 0;
	Tensor tensor;
	failure = Described(strict_text(entry), &tensor);
	float* floats = failure ? 0 : Read(path, &tensor, &failure);
	if (!floats)
		return strict_fail(failure);
	uint64_t total = tensor.height * tensor.width;
	free(storage);
	if (tensor.bytes > 1)
	{
		storage = malloc(sizeof(strict_packed) + total * sizeof(float));
		strict_packed* packed = (strict_packed*)storage;
		packed->count = (double)total;
		packed->bits = 32;
		memcpy(packed->data, floats, total * sizeof(float));
		free(floats);
		return strict_packed_result(packed);
	}
	strict_numbers* numbers = malloc(sizeof(strict_numbers) + total * sizeof(double));
	storage = (unsigned char*)numbers;
	numbers->count = (double)total;
	for (uint64_t index = 0; index < total; index++)
		numbers->items[index] = floats[index];
	free(floats);
	return strict_numbers_result(numbers);
}

/* The rows (or the columns when transposed) of the matrix that entry describes, each as numbers with bits 32 (exact for
   32 and 16 bit tensors), 16 (rounded to halves) or 8 (quantized, strict_quantized). */
static double Matrix(double path, double entry, double bits, int transposed)
{
	static double* rows = 0;
	static unsigned char* storage = 0;
	Tensor tensor;
	const char* failure = Described(strict_text(entry), &tensor);
	if (failure)
		return strict_fail(failure);
	if (tensor.rank != 2 || (bits != 8 && bits != 16 && bits != 32))
		return strict_fail("OnlyMatricesOf8Or16Or32BitsAreSupported");
	uint64_t height = tensor.height, width = tensor.width;
	float* floats = Read(path, &tensor, &failure);
	if (!floats)
		return strict_fail(failure);
	uint64_t outer = transposed ? width : height, inner = transposed ? height : width;
	uint64_t size = (sizeof(strict_packed) + strict_packed_bytes(inner, (uint64_t)bits) + 7) / 8 * 8;
	free(rows);
	free(storage);
	rows = malloc((1 + outer) * sizeof(double));
	storage = malloc(outer * size + 1);
	rows[0] = (double)outer;
	for (uint64_t row = 0; row < outer; row++)
	{
		strict_packed* packed = (strict_packed*)(storage + row * size);
		const float* first = transposed ? floats + row : floats + row * width;
		uint64_t stride = transposed ? width : 1;
		packed->count = (double)inner;
		packed->bits = bits;
		if (bits == 8)
			strict_quantized(packed, first, inner, stride);
		for (uint64_t column = 0; bits != 8 && column < inner; column++)
			if (bits == 16)
				((uint16_t*)packed->data)[column] = strict_half(first[column * stride]);
			else
				((float*)packed->data)[column] = first[column * stride];
		rows[1 + row] = strict_packed_result(packed);
	}
	free(floats);
	return strict_numbers_result((strict_numbers*)rows);
}

STRICT_EXPORT double TensorFile_Rows(double self, double path, double entry, double bits)
{
	return Matrix(path, entry, bits, 0);
}

STRICT_EXPORT double TensorFile_Columns(double self, double path, double entry, double bits)
{
	return Matrix(path, entry, bits, 1);
}

/* The rows of an I8 matrix with one F32 scale per row (comfy_quant int8 rows, convrot keeps them rotated), packed as 8 bit
   rows whose per-32 scales all are the scale of their row, so every number is exact. */
STRICT_EXPORT double TensorFile_Quantized(double self, double path, double entry, double scales)
{
	static double* rows = 0;
	static unsigned char* storage = 0;
	Tensor tensor, scaled;
	const char* failure = Described(strict_text(entry), &tensor);
	if (!failure)
		failure = Described(strict_text(scales), &scaled);
	if (!failure && (!tensor.isSigned || tensor.rank != 2 || scaled.bytes != 4 || scaled.height * scaled.width != tensor.height))
		failure = "OnlyI8MatricesWithAnF32ScalePerRowAreSupported";
	if (failure)
		return strict_fail(failure);
	float* values = Read(path, &tensor, &failure);
	float* factors = values ? Read(path, &scaled, &failure) : 0;
	if (!factors)
	{
		free(values);
		return strict_fail(failure);
	}
	uint64_t height = tensor.height, width = tensor.width;
	uint64_t size = (sizeof(strict_packed) + strict_packed_bytes(width, 8) + 7) / 8 * 8;
	free(rows);
	free(storage);
	rows = malloc((1 + height) * sizeof(double));
	storage = malloc(height * size + 1);
	rows[0] = (double)height;
	for (uint64_t row = 0; row < height; row++)
	{
		strict_packed* packed = (strict_packed*)(storage + row * size);
		float* parts = (float*)(packed->data + strict_packed_data(width, 8));
		packed->count = (double)width;
		packed->bits = 8;
		for (uint64_t column = 0; column < width; column++)
			((int8_t*)packed->data)[column] = (int8_t)values[row * width + column];
		for (uint64_t part = 0; part < (width + 31) / 32; part++)
			parts[part] = factors[row];
		rows[1 + row] = strict_packed_result(packed);
	}
	free(values);
	free(factors);
	return strict_numbers_result((strict_numbers*)rows);
}
