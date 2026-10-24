#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "strict.h"
#include "Parallel.h"
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
	uint64_t range[2];
	if (Listed(offsets + 16, range, 2) != 2)
		return "TensorOffsetsMalformed";
	tensor->rank = 0;
	tensor->width = 1;
	tensor->height = 1;
	char* dimension = (char*)shape + 9;
	while (*dimension != ']')
	{
		if (*dimension < '0' || *dimension > '9')
			return "TensorShapeMalformed";
		tensor->height *= tensor->width;
		tensor->width = strtoull(dimension, &dimension, 10);
		if (tensor->height > (1ull << 30) || tensor->width > (1ull << 30))
			return "TensorTooLarge";
		tensor->rank++;
		if (*dimension == ',')
			dimension++;
		else if (*dimension != ']')
			return "TensorShapeMalformed";
	}
	tensor->start = range[0];
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

/* The numbers of the tensor that entry describes: 32 bit floats for float tensors when bits is 32 (exact, the GPU keeps
   them as they are), doubles otherwise. */
STRICT_EXPORT double TensorFile_Floats(double self, double path, double entry, double bits)
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
	if (bits == 32 && tensor.bytes > 1)
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

/* The result of the last matrix call (the caller copies it right away): the row count, then every row as a packed list
   in storage, which is kept for the next call (fresh memory costs a page fault per 4 KB, 0.4 ms per MB). */
static double* rows = 0;
static unsigned char* storage = 0;
static uint64_t capacity = 0;

/* Room for height rows of size bytes each, 0 if there is not enough memory. */
static int Allocated(uint64_t height, uint64_t size)
{
	free(rows);
	rows = malloc((1 + height) * sizeof(double));
	if (height * size + 1 > capacity)
	{
		free(storage);
		storage = malloc(height * size + 1);
		capacity = storage ? height * size + 1 : 0;
	}
	if (rows)
		rows[0] = (double)height;
	return rows && storage;
}

enum { ChunkRows = 64 };
typedef struct { Opening file; uint64_t start, width, rowBytes, size, height; double bits; atomic_int failed; } Placing;

/* Reads the rows of one chunk at once into a buffer and copies them into their packed rows. */
static void PlacedChunk(void* context, uint64_t chunk)
{
	Placing* placing = context;
	uint64_t first = chunk * ChunkRows, count = placing->height - first < ChunkRows ? placing->height - first : ChunkRows;
	unsigned char* buffer = malloc(count * placing->rowBytes + 1);
	if (!buffer || !ReadFrom(placing->file, placing->start + first * placing->rowBytes, buffer, count * placing->rowBytes))
		atomic_store(&placing->failed, 1);
	for (uint64_t row = first; buffer && row < first + count; row++)
	{
		strict_packed* packed = (strict_packed*)(storage + row * placing->size);
		packed->count = (double)placing->width;
		packed->bits = placing->bits;
		memcpy(packed->data, buffer + (row - first) * placing->rowBytes, placing->rowBytes);
		rows[1 + row] = strict_packed_result(packed);
	}
	free(buffer);
}

/* Reads the rows of tensor as they are in the file straight into the packed rows (size bytes apart) with bits, in
   chunks on all threads, the failure or 0. */
static const char* Placed(double path, const Tensor* tensor, uint64_t size, double bits)
{
	uint64_t length = 0, fileSize = 0;
	FILE* file = Opened(strict_text(path), &length);
	if (!file)
		return "TensorFileNotReadable";
	fclose(file);
	Placing placing = { OpenedFile(strict_text(path), &fileSize), 8 + length + tensor->start, tensor->width,
		tensor->width * tensor->bytes, size, tensor->height, bits };
	if (!placing.file)
		return "TensorFileNotReadable";
	Parallel(PlacedChunk, &placing, (tensor->height + ChunkRows - 1) / ChunkRows);
	Closed(placing.file);
	return atomic_load(&placing.failed) ? "TensorFileDataTruncated" : 0;
}

/* The rows (or the columns when transposed) of the matrix that entry describes, each as numbers with bits 32 (exact for
   32 and 16 bit tensors), 16 (rounded to halves, F16 rows are read as they are) or 8 (quantized, strict_quantized). */
static double Matrix(double path, double entry, double bits, int transposed)
{
	Tensor tensor;
	const char* failure = Described(strict_text(entry), &tensor);
	if (failure)
		return strict_fail(failure);
	if (tensor.rank != 2 || (bits != 8 && bits != 16 && bits != 32))
		return strict_fail("OnlyMatricesOf8Or16Or32BitsAreSupported");
	uint64_t height = tensor.height, width = tensor.width;
	uint64_t outer = transposed ? width : height, inner = transposed ? height : width;
	uint64_t size = (sizeof(strict_packed) + strict_packed_bytes(inner, (uint64_t)bits) + 7) / 8 * 8;
	int isStored = !transposed && !tensor.isBrain && bits > 8 && tensor.bytes * 8 == bits;
	float* floats = isStored ? 0 : Read(path, &tensor, &failure);
	if (failure)
		return strict_fail(failure);
	failure = !Allocated(outer, size) ? "TensorTooLarge" : isStored ? Placed(path, &tensor, size, bits) : 0;
	for (uint64_t row = 0; floats && !failure && row < outer; row++)
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
	return failure ? strict_fail(failure) : strict_numbers_result((strict_numbers*)rows);
}

STRICT_EXPORT double TensorFile_Rows(double self, double path, double entry, double bits)
{
	return Matrix(path, entry, bits, 0);
}

STRICT_EXPORT double TensorFile_Columns(double self, double path, double entry, double bits)
{
	return Matrix(path, entry, bits, 1);
}

/* The rows of an I8 matrix with one F32 scale per row (comfy_quant int8 rows, convrot keeps them rotated), read straight
   into 8 bit rows whose per-32 scales all are the scale of their row, so every number is exact. */
STRICT_EXPORT double TensorFile_Quantized(double self, double path, double entry, double scales)
{
	Tensor tensor, scaled;
	const char* failure = Described(strict_text(entry), &tensor);
	if (!failure)
		failure = Described(strict_text(scales), &scaled);
	if (!failure && (!tensor.isSigned || tensor.rank != 2 || scaled.bytes != 4 || scaled.height * scaled.width != tensor.height))
		failure = "OnlyI8MatricesWithAnF32ScalePerRowAreSupported";
	float* factors = failure ? 0 : Read(path, &scaled, &failure);
	if (!factors)
		return strict_fail(failure);
	uint64_t width = tensor.width, size = (sizeof(strict_packed) + strict_packed_bytes(width, 8) + 7) / 8 * 8;
	failure = Allocated(tensor.height, size) ? Placed(path, &tensor, size, 8) : "TensorTooLarge";
	for (uint64_t row = 0; !failure && row < tensor.height; row++)
	{
		float* parts = (float*)(storage + row * size + sizeof(strict_packed) + strict_packed_data(width, 8));
		for (uint64_t part = 0; part < (width + 31) / 32; part++)
			parts[part] = factors[row];
	}
	free(factors);
	return failure ? strict_fail(failure) : strict_numbers_result((strict_numbers*)rows);
}
