#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "strict.h"
#ifdef _WIN32
#define strict_seek _fseeki64
#define strict_tell _ftelli64
#else
#define strict_seek fseeko
#define strict_tell ftello
#endif

/* A GGUF file (llama.cpp models): the metadata (keys with numbers, texts or arrays of them) and the tensor infos are read
   once per path, tensors when asked for. Supported tensor types are F32, F16, BF16 and Q8_0. */
enum { F32 = 0, F16 = 1, Q8_0 = 8, BF16 = 30 };
typedef struct { char* key; char* value; } Entry;
typedef struct { char* name; uint32_t type; uint32_t dimensions; uint64_t shape[4]; uint64_t offset; } Tensor;
typedef struct { FILE* file; int failed; } Reader;
typedef struct { char* data; uint64_t length; uint64_t capacity; } Builder;
static char* loadedPath = 0;
static Entry* entries = 0;
static uint64_t entryCount = 0;
static Tensor* tensors = 0;
static uint64_t tensorCount = 0;
static uint64_t dataStart = 0;

static void Read(Reader* reader, void* target, uint64_t size)
{
	if (!reader->failed && fread(target, 1, size, reader->file) != size)
		reader->failed = 1;
}

static uint64_t Unsigned(Reader* reader, int bytes)
{
	uint64_t value = 0;
	unsigned char data[8] = { 0 };
	Read(reader, data, bytes);
	for (int index = bytes - 1; index >= 0; index--)
		value = value * 256 + data[index];
	return value;
}

static char* Text(Reader* reader)
{
	uint64_t length = Unsigned(reader, 8);
	if (reader->failed || length > 100000000)
	{
		reader->failed = 1;
		return 0;
	}
	char* text = malloc(length + 1);
	Read(reader, text, length);
	text[length] = 0;
	return text;
}

/* Appends one scalar of the GGUF value type as text (floats like printf %.9g), returns 0 for an unknown type. */
static int Appended(Reader* reader, uint32_t type, Builder* text)
{
	static const int sizes[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
	char number[64];
	char* added = number;
	if (type == 8)
		added = Text(reader);
	else if (type > 12 || type == 9)
		return 0;
	else
	{
		uint64_t bits = Unsigned(reader, sizes[type]);
		int shift = 64 - 8 * sizes[type];
		union { uint32_t bits; float value; } single = { (uint32_t)bits };
		if (type == 6)
			snprintf(number, sizeof number, "%.9g", single.value);
		else if (type == 12)
			snprintf(number, sizeof number, "%.17g", strict_value(bits));
		else if (type == 7)
			snprintf(number, sizeof number, "%s", bits ? "true" : "false");
		else if (type == 1 || type == 3 || type == 5 || type == 11)
			snprintf(number, sizeof number, "%lld", (long long)((int64_t)(bits << shift) >> shift));
		else
			snprintf(number, sizeof number, "%llu", (unsigned long long)bits);
	}
	if (!added)
		return 0;
	uint64_t size = strlen(added);
	if (text->length + size + 2 > text->capacity)
	{
		text->capacity = (text->length + size + 2) * 2;
		text->data = realloc(text->data, text->capacity);
	}
	memcpy(text->data + text->length, added, size + 1);
	text->length += size;
	if (added != number)
		free(added);
	return 1;
}

/* The value of one metadata entry as text, an array as its elements one per line. */
static char* Value(Reader* reader)
{
	uint32_t type = (uint32_t)Unsigned(reader, 4);
	Builder text = { calloc(1, 16), 0, 16 };
	uint32_t itemType = type == 9 ? (uint32_t)Unsigned(reader, 4) : type;
	uint64_t count = type == 9 ? Unsigned(reader, 8) : 1;
	for (uint64_t index = 0; index < count && !reader->failed; index++)
	{
		if (index > 0)
			text.data[text.length++] = 10;
		if (!Appended(reader, itemType, &text))
		{
			free(text.data);
			return 0;
		}
	}
	return text.data;
}

static void Forget(void)
{
	for (uint64_t index = 0; index < entryCount; index++)
	{
		free(entries[index].key);
		free(entries[index].value);
	}
	for (uint64_t index = 0; index < tensorCount; index++)
		free(tensors[index].name);
	free(entries);
	free(tensors);
	free(loadedPath);
	entries = 0;
	tensors = 0;
	loadedPath = 0;
	entryCount = tensorCount = 0;
}

/* Reads the header of the file at path unless it is the one already loaded, returns the failure name if it can't. */
static const char* Loaded(const char* path)
{
	if (loadedPath && strcmp(loadedPath, path) == 0)
		return 0;
	Forget();
	Reader reader = { fopen(path, "rb"), 0 };
	if (!reader.file)
		return "GgufFileNotReadable";
	uint32_t magic = (uint32_t)Unsigned(&reader, 4);
	uint32_t version = (uint32_t)Unsigned(&reader, 4);
	uint64_t declaredTensors = Unsigned(&reader, 8);
	uint64_t declaredEntries = Unsigned(&reader, 8);
	if (reader.failed || magic != 0x46554747 || version < 2 || declaredTensors > 1000000 || declaredEntries > 1000000)
	{
		fclose(reader.file);
		return "NotAGgufFile";
	}
	uint64_t alignment = 32;
	entries = calloc(declaredEntries + 1, sizeof(Entry));
	tensors = calloc(declaredTensors + 1, sizeof(Tensor));
	for (; entryCount < declaredEntries && !reader.failed; entryCount++)
	{
		entries[entryCount].key = Text(&reader);
		entries[entryCount].value = entries[entryCount].key ? Value(&reader) : 0;
		if (!entries[entryCount].value)
			reader.failed = 1;
		else if (strcmp(entries[entryCount].key, "general.alignment") == 0)
			alignment = strtoull(entries[entryCount].value, 0, 10);
	}
	for (; tensorCount < declaredTensors && !reader.failed; tensorCount++)
	{
		Tensor* tensor = &tensors[tensorCount];
		tensor->name = Text(&reader);
		tensor->dimensions = (uint32_t)Unsigned(&reader, 4);
		if (!tensor->name || tensor->dimensions > 4)
			reader.failed = 1;
		for (uint32_t dimension = 0; dimension < tensor->dimensions && !reader.failed; dimension++)
			tensor->shape[dimension] = Unsigned(&reader, 8);
		tensor->type = (uint32_t)Unsigned(&reader, 4);
		tensor->offset = Unsigned(&reader, 8);
	}
	uint64_t end = (uint64_t)strict_tell(reader.file);
	fclose(reader.file);
	if (reader.failed || alignment == 0)
	{
		Forget();
		return "GgufHeaderTruncated";
	}
	dataStart = (end + alignment - 1) / alignment * alignment;
	loadedPath = malloc(strlen(path) + 1);
	strcpy(loadedPath, path);
	return 0;
}

static const Tensor* Found(const char* name)
{
	for (uint64_t index = 0; index < tensorCount; index++)
		if (strcmp(tensors[index].name, name) == 0)
			return &tensors[index];
	return 0;
}

/* The metadata value of key as text (an array one element per line), an empty text when the file has no such key. */
STRICT_EXPORT double GgufFile_Value(double self, double path, double key)
{
	const char* failure = Loaded(strict_text(path));
	if (failure)
		return strict_fail(failure);
	for (uint64_t index = 0; index < entryCount; index++)
		if (strcmp(entries[index].key, strict_text(key)) == 0)
			return strict_text_result(entries[index].value);
	return strict_text_result("");
}

/* The shape of the tensor name with the slowest changing dimension first (rows, columns), empty without such a tensor. */
STRICT_EXPORT double GgufFile_Shape(double self, double path, double name)
{
	static strict_numbers* shape = 0;
	const char* failure = Loaded(strict_text(path));
	if (failure)
		return strict_fail(failure);
	const Tensor* tensor = Found(strict_text(name));
	uint32_t dimensions = tensor ? tensor->dimensions : 0;
	free(shape);
	shape = malloc(sizeof(strict_numbers) + 4 * sizeof(double));
	shape->count = dimensions;
	for (uint32_t index = 0; index < dimensions; index++)
		shape->items[index] = (double)tensor->shape[dimensions - 1 - index];
	return strict_numbers_result(shape);
}

static double Half(uint16_t half)
{
	uint32_t sign = (half & 0x8000u) << 16, exponent = (half >> 10) & 31, mantissa = half & 1023;
	union { uint32_t bits; float value; } single;
	if (exponent == 0)
		return (sign ? -1.0 : 1.0) * mantissa / 16777216.0;
	single.bits = sign | (exponent == 31 ? 0x7F800000u : (exponent + 112) << 23) | mantissa << 13;
	return single.value;
}

/* Reads all numbers of the tensor as floats, 0 and the failure name if it can't. */
static float* Floats(const char* path, const Tensor* tensor, uint64_t count, const char** failure)
{
	uint64_t elementBytes = tensor->type == F32 ? 4 : tensor->type == Q8_0 ? 34 : 2;
	uint64_t elements = tensor->type == Q8_0 ? count / 32 : count;
	if ((tensor->type != F32 && tensor->type != F16 && tensor->type != BF16 && tensor->type != Q8_0) ||
		(tensor->type == Q8_0 && count % 32 != 0))
	{
		*failure = "OnlyF32F16BF16AndQ8_0TensorsAreSupported";
		return 0;
	}
	FILE* file = fopen(path, "rb");
	unsigned char* raw = malloc(elements * elementBytes + 1);
	float* floats = malloc(count * sizeof(float) + 1);
	size_t read = 0;
	if (file && strict_seek(file, (int64_t)(dataStart + tensor->offset), SEEK_SET) == 0)
		read = fread(raw, elementBytes, elements, file);
	if (file)
		fclose(file);
	for (uint64_t index = 0; read == elements && index < count; index++)
		if (tensor->type == F32)
			memcpy(&floats[index], raw + 4 * index, 4);
		else if (tensor->type == F16)
			floats[index] = (float)Half((uint16_t)(raw[2 * index] | raw[2 * index + 1] << 8));
		else if (tensor->type == BF16)
		{
			union { uint32_t bits; float value; } single = { (uint32_t)(raw[2 * index] | raw[2 * index + 1] << 8) << 16 };
			floats[index] = single.value;
		}
		else
		{
			const unsigned char* block = raw + index / 32 * 34;
			floats[index] = (float)(Half((uint16_t)(block[0] | block[1] << 8)) * (int8_t)block[2 + index % 32]);
		}
	free(raw);
	if (read == elements)
		return floats;
	free(floats);
	*failure = "GgufTensorDataTruncated";
	return 0;
}

static const Tensor* Requested(double path, double name, uint32_t dimensions, const char** failure)
{
	*failure = Loaded(strict_text(path));
	const Tensor* tensor = *failure ? 0 : Found(strict_text(name));
	if (!*failure && (!tensor || (dimensions && tensor->dimensions != dimensions)))
		*failure = "GgufTensorNotFound";
	return *failure ? 0 : tensor;
}

static uint64_t Elements(const Tensor* tensor)
{
	uint64_t elements = 1;
	for (uint32_t index = 0; index < tensor->dimensions; index++)
		elements *= tensor->shape[index];
	return elements;
}

/* All numbers of the tensor name as doubles. */
STRICT_EXPORT double GgufFile_Values(double self, double path, double name)
{
	static strict_numbers* numbers = 0;
	const char* failure = 0;
	const Tensor* tensor = Requested(path, name, 0, &failure);
	uint64_t count = tensor ? Elements(tensor) : 0;
	float* floats = tensor ? Floats(strict_text(path), tensor, count, &failure) : 0;
	if (!floats)
		return strict_fail(failure);
	free(numbers);
	numbers = malloc(sizeof(strict_numbers) + count * sizeof(double));
	numbers->count = (double)count;
	for (uint64_t index = 0; index < count; index++)
		numbers->items[index] = floats[index];
	free(floats);
	return strict_numbers_result(numbers);
}

/* The rows of the matrix name (its contiguous dimension is a row) packed with bits 32, 16 or 8. Q8_0 rows packed with 8
   bits keep their numbers: their scales are halves, so strict_quantized finds the same scales and bytes again. */
STRICT_EXPORT double GgufFile_Rows(double self, double path, double name, double bits)
{
	static double* rows = 0;
	static unsigned char* storage = 0;
	const char* failure = bits != 8 && bits != 16 && bits != 32 ? "OnlyMatricesOf8Or16Or32BitsAreSupported" : 0;
	const Tensor* tensor = failure ? 0 : Requested(path, name, 2, &failure);
	uint64_t width = tensor ? tensor->shape[0] : 0, height = tensor ? tensor->shape[1] : 0;
	float* floats = tensor ? Floats(strict_text(path), tensor, width * height, &failure) : 0;
	if (!floats)
		return strict_fail(failure);
	uint64_t size = (sizeof(strict_packed) + strict_packed_bytes(width, (uint64_t)bits) + 7) / 8 * 8;
	free(rows);
	free(storage);
	rows = malloc((1 + height) * sizeof(double));
	storage = malloc(height * size + 1);
	rows[0] = (double)height;
	for (uint64_t row = 0; row < height; row++)
	{
		strict_packed* packed = (strict_packed*)(storage + row * size);
		const float* first = floats + row * width;
		packed->count = (double)width;
		packed->bits = bits;
		if (bits == 8)
			strict_quantized(packed, first, width, 1);
		for (uint64_t column = 0; bits != 8 && column < width; column++)
			if (bits == 16)
				((uint16_t*)packed->data)[column] = strict_half(first[column]);
			else
				((float*)packed->data)[column] = first[column];
		rows[1 + row] = strict_packed_result(packed);
	}
	free(floats);
	return strict_numbers_result((strict_numbers*)rows);
}
