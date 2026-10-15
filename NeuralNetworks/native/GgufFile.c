#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "strict.h"
#pragma STDC FP_CONTRACT OFF
#ifdef _WIN32
#define strict_seek _fseeki64
#define strict_tell _ftelli64
#else
#define strict_seek fseeko
#define strict_tell ftello
#endif

/* A GGUF file (llama.cpp models): the metadata (keys with numbers, texts or arrays of them) and the tensor infos are read
   once per path, tensors when asked for. Supported tensor types are F32, F16, BF16, Q8_0, Q4_K and Q6_K, blocks decode to
   floats exactly like ggml's dequantize_row functions (no fused multiply-add). */
enum { F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q6_K = 14, BF16 = 30 };
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

static uint64_t BlockNumbers(uint32_t type) { return type == Q8_0 ? 32 : type == Q4_K || type == Q6_K ? 256 : 1; }
static uint64_t BlockBytes(uint32_t type) { return type == F32 ? 4 : type == Q8_0 ? 34 : type == Q4_K ? 144 : type == Q6_K ? 210 : 2; }
static float HalfAt(const unsigned char* bytes) { return (float)Half((uint16_t)(bytes[0] | bytes[1] << 8)); }

/* The 6 bit scale and minimum of group j of a Q4_K block (get_scale_min_k4 of ggml). */
static void ScaleMin(int j, const unsigned char* q, int* scale, int* minimum)
{
	*scale = j < 4 ? q[j] & 63 : (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
	*minimum = j < 4 ? q[j + 4] & 63 : (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}

/* 256 numbers of 4 bits with a scale and minimum per 32 (d, dmin, 12 bytes of scales, 128 bytes of nibbles). */
static void DecodedQ4K(const unsigned char* block, float* y)
{
	float d = HalfAt(block), low = HalfAt(block + 2);
	const unsigned char* q = block + 16;
	for (int first = 0; first < 256; first += 64, q += 32)
	{
		int scale, minimum;
		ScaleMin(first / 32, block + 4, &scale, &minimum);
		float d1 = d * scale, m1 = low * minimum;
		ScaleMin(first / 32 + 1, block + 4, &scale, &minimum);
		float d2 = d * scale, m2 = low * minimum;
		for (int index = 0; index < 32; index++)
		{
			y[first + index] = d1 * (q[index] & 0xF) - m1;
			y[first + 32 + index] = d2 * (q[index] >> 4) - m2;
		}
	}
}

/* 256 numbers of 6 bits (low nibbles, high bit pairs, a signed scale per 16 and d). */
static void DecodedQ6K(const unsigned char* block, float* y)
{
	const unsigned char* low = block;
	const unsigned char* high = block + 128;
	const int8_t* scales = (const int8_t*)(block + 192);
	float d = HalfAt(block + 208);
	for (int first = 0; first < 256; first += 128, low += 64, high += 32, scales += 8)
		for (int index = 0; index < 32; index++)
		{
			int group = index / 16;
			y[first + index] = d * scales[group] * (((low[index] & 0xF) | ((high[index] & 3) << 4)) - 32);
			y[first + index + 32] = d * scales[group + 2] * (((low[index + 32] & 0xF) | (((high[index] >> 2) & 3) << 4)) - 32);
			y[first + index + 64] = d * scales[group + 4] * (((low[index] >> 4) | (((high[index] >> 4) & 3) << 4)) - 32);
			y[first + index + 96] = d * scales[group + 6] * (((low[index + 32] >> 4) | (((high[index] >> 6) & 3) << 4)) - 32);
		}
}

/* Decodes one block of the tensor's type into floats. */
static void Decoded(uint32_t type, const unsigned char* block, float* y)
{
	if (type == F32)
		memcpy(y, block, 4);
	else if (type == F16)
		*y = HalfAt(block);
	else if (type == BF16)
	{
		union { uint32_t bits; float value; } single = { (uint32_t)(block[0] | block[1] << 8) << 16 };
		*y = single.value;
	}
	else if (type == Q8_0)
		for (int index = 0; index < 32; index++)
			y[index] = (float)(Half((uint16_t)(block[0] | block[1] << 8)) * (int8_t)block[2 + index]);
	else if (type == Q4_K)
		DecodedQ4K(block, y);
	else
		DecodedQ6K(block, y);
}

/* Reads the blocks of the tensor as they are stored, 0 and the failure name if it can't. */
static unsigned char* Blocks(const char* path, const Tensor* tensor, uint64_t count, const char** failure)
{
	uint32_t type = tensor->type;
	uint64_t blockBytes = BlockBytes(type), blocks = count / BlockNumbers(type);
	if ((type != F32 && type != F16 && type != BF16 && type != Q8_0 && type != Q4_K && type != Q6_K) ||
		count % BlockNumbers(type) != 0)
	{
		*failure = "OnlyF32F16BF16Q8_0Q4_KAndQ6_KTensorsAreSupported";
		return 0;
	}
	FILE* file = fopen(path, "rb");
	unsigned char* raw = malloc(blocks * blockBytes + 1);
	size_t read = 0;
	if (file && strict_seek(file, (int64_t)(dataStart + tensor->offset), SEEK_SET) == 0)
		read = fread(raw, blockBytes, blocks, file);
	if (file)
		fclose(file);
	if (read == blocks)
		return raw;
	free(raw);
	*failure = "GgufTensorDataTruncated";
	return 0;
}

/* Reads all numbers of the tensor as floats, 0 and the failure name if it can't. */
static float* Floats(const char* path, const Tensor* tensor, uint64_t count, const char** failure)
{
	unsigned char* raw = Blocks(path, tensor, count, failure);
	float* floats = raw ? malloc(count * sizeof(float) + 1) : 0;
	uint64_t blockNumbers = BlockNumbers(tensor->type), blockBytes = BlockBytes(tensor->type);
	for (uint64_t block = 0; floats && block < count / blockNumbers; block++)
		Decoded(tensor->type, raw + block * blockBytes, floats + block * blockNumbers);
	free(raw);
	return floats;
}

/* A Q4_K block as 256 nibbles in order and a float scale and minimum for each 32 (packed rows of 4 bits). */
static void PackedQ4K(const unsigned char* block, unsigned char* nibbles, float* pairs)
{
	float d = HalfAt(block), low = HalfAt(block + 2);
	for (int group = 0; group < 8; group++)
	{
		int scale, minimum;
		ScaleMin(group, block + 4, &scale, &minimum);
		pairs[2 * group] = d * scale;
		pairs[2 * group + 1] = low * minimum;
	}
	for (int index = 0; index < 256; index += 2)
	{
		const unsigned char* q = block + 16 + index / 64 * 32 + index % 32;
		int high = index % 64 >= 32;
		int first = high ? q[0] >> 4 : q[0] & 0xF, second = high ? q[1] >> 4 : q[1] & 0xF;
		nibbles[index / 2] = (unsigned char)(first | second << 4);
	}
}

/* A Q6_K block as 256 signed bytes in order and a float scale for each 16 (packed rows of 6 bits). */
static void PackedQ6K(const unsigned char* block, int8_t* bytes, float* scales)
{
	float values[256];
	float d = HalfAt(block + 208);
	DecodedQ6K(block, values);
	for (int group = 0; group < 16; group++)
		scales[group] = d * ((const int8_t*)(block + 192))[group];
	for (int index = 0; index < 256; index++)
		bytes[index] = scales[index / 16] == 0 ? 0 : (int8_t)(values[index] / scales[index / 16]);
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
/* The blocks of one row of a Q4_K or Q6_K matrix as a packed row of 4 or 6 bits. */
static void Repacked(const unsigned char* blocks, uint64_t bits, uint64_t width, strict_packed* packed)
{
	float* scales = (float*)(packed->data + strict_packed_data(width, bits));
	for (uint64_t block = 0; block < width / 256; block++)
		if (bits == 4)
			PackedQ4K(blocks + block * 144, packed->data + block * 128, scales + block * 16);
		else
			PackedQ6K(blocks + block * 210, (int8_t*)packed->data + block * 256, scales + block * 16);
}

/* Rows of 32, 16 or 8 bits, 4 keeps Q4_K and Q6_K matrices in 4 and 6 bits (other types become 8 bits). The matrices of
   a stack (the experts of a mixture) follow each other. */
STRICT_EXPORT double GgufFile_Rows(double self, double path, double name, double bits)
{
	static double* rows = 0;
	static unsigned char* storage = 0;
	const char* failure = bits != 4 && bits != 8 && bits != 16 && bits != 32 ? "OnlyMatricesOf4Or8Or16Or32BitsAreSupported" : 0;
	const Tensor* tensor = failure ? 0 : Requested(path, name, 0, &failure);
	if (tensor && tensor->dimensions != 2 && tensor->dimensions != 3)
		tensor = 0, failure = "OnlyMatricesAndStacksOfThemHaveRows";
	uint64_t width = tensor ? tensor->shape[0] : 0;
	uint64_t height = tensor ? tensor->shape[1] * (tensor->dimensions == 3 ? tensor->shape[2] : 1) : 0;
	uint32_t type = tensor ? tensor->type : F32;
	uint64_t kept = bits != 4 ? (uint64_t)bits : type == Q4_K ? 4 : type == Q6_K ? 6 : 8;
	int isBlocked = kept == 4 || kept == 6;
	unsigned char* raw = tensor && isBlocked ? Blocks(strict_text(path), tensor, width * height, &failure) : 0;
	float* floats = tensor && !isBlocked ? Floats(strict_text(path), tensor, width * height, &failure) : 0;
	if (!floats && !raw)
		return strict_fail(failure);
	uint64_t size = (sizeof(strict_packed) + strict_packed_bytes(width, kept) + 7) / 8 * 8;
	free(rows);
	free(storage);
	rows = malloc((1 + height) * sizeof(double));
	storage = malloc(height * size + 1);
	rows[0] = (double)height;
	for (uint64_t row = 0; row < height; row++)
	{
		strict_packed* packed = (strict_packed*)(storage + row * size);
		packed->count = (double)width;
		packed->bits = (double)kept;
		if (isBlocked)
			Repacked(raw + row * (width / 256) * BlockBytes(type), kept, width, packed);
		else if (kept == 8)
			strict_quantized(packed, floats + row * width, width, 1);
		for (uint64_t column = 0; floats && kept != 8 && column < width; column++)
			if (kept == 16)
				((uint16_t*)packed->data)[column] = strict_half(floats[row * width + column]);
			else
				((float*)packed->data)[column] = floats[row * width + column];
		rows[1 + row] = strict_packed_result(packed);
	}
	free(floats);
	free(raw);
	return strict_numbers_result((strict_numbers*)rows);
}
