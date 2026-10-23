#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdatomic.h>
#include "strict.h"
#include "ggml-cpu/quants.h"
#include "ggml-impl.h"
#pragma STDC FP_CONTRACT OFF
#ifdef _WIN32
#include <windows.h>
#define strict_tell _ftelli64
#else
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#define strict_tell ftello
#endif

/* A GGUF file (llama.cpp models): the metadata (keys with numbers, texts or arrays of them) and the tensor infos are read
   once per path, tensors when asked for. Supported tensor types are F32, F16, BF16, Q8_0, Q5_1, Q4_K, Q5_K, Q6_K and
   IQ4_NL, blocks decode to floats exactly like ggml's dequantize_row functions (no fused multiply-add). */
enum { F32 = 0, F16 = 1, Q5_1 = 7, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ4_NL = 20, BF16 = 30 };
typedef struct { char* key; char* value; } Entry;
typedef struct { char* name; uint32_t type; uint32_t dimensions; uint64_t shape[4]; uint64_t offset; uint32_t shard; uint64_t start; } Tensor;
typedef struct { FILE* file; int failed; } Reader;
typedef struct { char* data; uint64_t length; uint64_t capacity; } Builder;
static char* loadedPath = 0;
static Entry* entries = 0;
static uint64_t entryCount = 0;
static Tensor* tensors = 0;
static uint64_t tensorCount = 0;
static char** shardPaths = 0;
static uint32_t shardCount = 0;

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

#ifdef _WIN32
typedef HANDLE Opening;
/* The file at path opened for reads at any offset from any thread, its size, 0 if it can't be opened. */
static Opening OpenedFile(const char* path, uint64_t* size)
{
	LARGE_INTEGER length;
	HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, 0);
	if (file == INVALID_HANDLE_VALUE)
		return 0;
	*size = GetFileSizeEx(file, &length) ? (uint64_t)length.QuadPart : 0;
	return file;
}

static void Closed(Opening file) { CloseHandle(file); }

/* Reads bytes at offset of the file into target, threads read at the same time, in parts of at most 1 GB. */
static int ReadFrom(Opening file, uint64_t offset, unsigned char* target, uint64_t bytes)
{
	int isRead = 1;
	HANDLE event = CreateEventA(0, TRUE, FALSE, 0);
	for (uint64_t done = 0; isRead && done < bytes;)
	{
		DWORD part = bytes - done > 1073741824 ? 1073741824 : (DWORD)(bytes - done), read = 0;
		OVERLAPPED overlapped = { 0 };
		overlapped.Offset = (DWORD)(offset + done);
		overlapped.OffsetHigh = (DWORD)((offset + done) >> 32);
		overlapped.hEvent = event;
		int isStarted = ReadFile(file, target + done, part, 0, &overlapped) || GetLastError() == ERROR_IO_PENDING;
		isRead = isStarted && GetOverlappedResult(file, &overlapped, &read, TRUE) && read > 0;
		done += read;
	}
	CloseHandle(event);
	return isRead;
}
#else
typedef intptr_t Opening;
/* The file at path opened for reads (its descriptor + 1), its size, 0 if it can't be opened. */
static Opening OpenedFile(const char* path, uint64_t* size)
{
	int file = open(path, O_RDONLY);
	if (file < 0)
		return 0;
	off_t length = lseek(file, 0, SEEK_END);
	*size = length > 0 ? (uint64_t)length : 0;
	return file + 1;
}

static void Closed(Opening file) { close((int)file - 1); }

static int ReadFrom(Opening file, uint64_t offset, unsigned char* target, uint64_t bytes)
{
	ssize_t read = 1;
	for (uint64_t done = 0; read > 0 && done < bytes; done += (uint64_t)read)
		read = pread((int)file - 1, target + done, bytes - done, (off_t)(offset + done));
	return read > 0;
}
#endif
static Opening* openings = 0;
static uint64_t* openingSizes = 0;

static void Forget(void)
{
	for (uint64_t index = 0; index < entryCount; index++)
	{
		free(entries[index].key);
		free(entries[index].value);
	}
	for (uint64_t index = 0; index < tensorCount; index++)
		free(tensors[index].name);
	for (uint32_t shard = 0; shard < shardCount; shard++)
	{
		free(shardPaths[shard]);
		if (openings && openings[shard])
			Closed(openings[shard]);
	}
	free(openings);
	free(openingSizes);
	openings = 0;
	openingSizes = 0;
	free(entries);
	free(tensors);
	free(shardPaths);
	free(loadedPath);
	entries = 0;
	tensors = 0;
	shardPaths = 0;
	loadedPath = 0;
	entryCount = tensorCount = 0;
	shardCount = 0;
}

/* Adds the metadata and tensor infos of one file of a model to the loaded ones, returns the failure name if it can't. */
static const char* Header(const char* path, uint32_t shard)
{
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
	uint64_t alignment = 32, firstTensor = tensorCount;
	entries = realloc(entries, (entryCount + declaredEntries + 1) * sizeof(Entry));
	tensors = realloc(tensors, (tensorCount + declaredTensors + 1) * sizeof(Tensor));
	memset(entries + entryCount, 0, (declaredEntries + 1) * sizeof(Entry));
	memset(tensors + tensorCount, 0, (declaredTensors + 1) * sizeof(Tensor));
	for (uint64_t index = 0; index < declaredEntries && !reader.failed; index++, entryCount++)
	{
		entries[entryCount].key = Text(&reader);
		entries[entryCount].value = entries[entryCount].key ? Value(&reader) : 0;
		if (!entries[entryCount].value)
			reader.failed = 1;
		else if (strcmp(entries[entryCount].key, "general.alignment") == 0)
			alignment = strtoull(entries[entryCount].value, 0, 10);
	}
	for (uint64_t index = 0; index < declaredTensors && !reader.failed; index++, tensorCount++)
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
		tensor->shard = shard;
	}
	uint64_t end = (uint64_t)strict_tell(reader.file);
	fclose(reader.file);
	if (reader.failed || alignment == 0)
		return "GgufHeaderTruncated";
	for (uint64_t index = firstTensor; index < tensorCount; index++)
		tensors[index].start = (end + alignment - 1) / alignment * alignment;
	return 0;
}

/* The name of file shard (from 0) of a model split like llama.cpp does (Name-00001-of-00004.gguf), 0 if path is not
   the first of such files. */
static char* ShardPath(const char* path, uint32_t shard)
{
	const char* marker = strstr(path, "-00001-of-");
	if (!marker)
		return 0;
	char* named = malloc(strlen(path) + 1);
	char number[8];
	strcpy(named, path);
	snprintf(number, sizeof(number), "%05u", shard + 1);
	memcpy(named + (marker - path) + 1, number, 5);
	return named;
}

/* Reads the header of the file at path unless it is the one already loaded, and of the other files of a split model
   (split.count in the first), returns the failure name if it can't. */
static const char* Loaded(const char* path)
{
	if (loadedPath && strcmp(loadedPath, path) == 0)
		return 0;
	Forget();
	const char* failure = Header(path, 0);
	uint32_t count = 1;
	for (uint64_t index = 0; !failure && index < entryCount; index++)
		if (strcmp(entries[index].key, "split.count") == 0)
			count = (uint32_t)strtoul(entries[index].value, 0, 10);
	shardPaths = calloc(count + 1, sizeof(char*));
	shardPaths[0] = malloc(strlen(path) + 1);
	strcpy(shardPaths[0], path);
	shardCount = 1;
	for (uint32_t shard = 1; !failure && shard < count; shard++)
	{
		shardPaths[shard] = ShardPath(path, shard);
		shardCount++;
		failure = shardPaths[shard] ? Header(shardPaths[shard], shard) : "GgufSplitFileNameNotFound";
	}
	if (failure)
	{
		Forget();
		return failure;
	}
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


static uint64_t BlockNumbers(uint32_t type)
{
	return type == Q8_0 || type == Q5_1 || type == IQ4_NL ? 32 : type == Q4_K || type == Q5_K || type == Q6_K ? 256 : 1;
}

/* The bytes of one block of the type, 0 if the type is not supported. */
static uint64_t BlockBytes(uint32_t type)
{
	switch (type)
	{
	case F32: return 4;
	case F16: case BF16: return 2;
	case Q5_1: return 24;
	case Q8_0: return 34;
	case Q4_K: return 144;
	case Q5_K: return 176;
	case Q6_K: return 210;
	case IQ4_NL: return 18;
	default: return 0;
	}
}
static float HalfAt(const unsigned char* bytes) { return strict_half_float((uint16_t)(bytes[0] | bytes[1] << 8)); }

/* The 6 bit scale and minimum of group j of a Q4_K block (get_scale_min_k4 of ggml). */
static void ScaleMin(int j, const unsigned char* q, int* scale, int* minimum)
{
	*scale = j < 4 ? q[j] & 63 : (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
	*minimum = j < 4 ? q[j + 4] & 63 : (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}

/* 256 numbers of 4 bits with a scale and minimum per 32 (d, dmin, 12 bytes of scales, 128 bytes of nibbles), Q5_K
   blocks have 32 bytes of fifth bits (bit j of byte index is number j * 32 + index) before the nibbles. */
static void DecodedQ4KOrQ5K(const unsigned char* block, int hasFifthBits, float* y)
{
	float d = HalfAt(block), low = HalfAt(block + 2);
	const unsigned char* high = block + 16;
	const unsigned char* q = block + (hasFifthBits ? 48 : 16);
	for (int first = 0; first < 256; first += 64, q += 32)
	{
		int scale, minimum;
		ScaleMin(first / 32, block + 4, &scale, &minimum);
		float d1 = d * scale, m1 = low * minimum;
		ScaleMin(first / 32 + 1, block + 4, &scale, &minimum);
		float d2 = d * scale, m2 = low * minimum;
		for (int index = 0; index < 32; index++)
		{
			int fifth = hasFifthBits ? high[index] >> (first / 32) : 0;
			y[first + index] = d1 * ((q[index] & 0xF) | (fifth & 1) << 4) - m1;
			y[first + 32 + index] = d2 * ((q[index] >> 4) | (fifth & 2) << 3) - m2;
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

/* 32 numbers of 5 bits: half d and minimum m, 4 bytes of fifth bits and 16 bytes of nibbles (low ones first). */
static void DecodedQ51(const unsigned char* block, float* y)
{
	float d = HalfAt(block), m = HalfAt(block + 2);
	uint32_t high = (uint32_t)block[4] | (uint32_t)block[5] << 8 | (uint32_t)block[6] << 16 | (uint32_t)block[7] << 24;
	for (int index = 0; index < 16; index++)
	{
		y[index] = ((block[8 + index] & 0xF) | ((high >> index) & 1) << 4) * d + m;
		y[index + 16] = ((block[8 + index] >> 4) | ((high >> (index + 16)) & 1) << 4) * d + m;
	}
}

/* 32 numbers of 4 bit indices into the non-linear kvalues_iq4nl of ggml times half d (low nibbles first). */
static void DecodedIQ4NL(const unsigned char* block, float* y)
{
	static const int8_t values[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
	float d = HalfAt(block);
	for (int index = 0; index < 16; index++)
	{
		y[index] = d * values[block[2 + index] & 0xF];
		y[index + 16] = d * values[block[2 + index] >> 4];
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
		*y = strict_brain_float((uint16_t)(block[0] | block[1] << 8));
	else if (type == Q8_0)
		for (int index = 0; index < 32; index++)
			y[index] = HalfAt(block) * (int8_t)block[2 + index];
	else if (type == Q5_1)
		DecodedQ51(block, y);
	else if (type == Q4_K || type == Q5_K)
		DecodedQ4KOrQ5K(block, type == Q5_K, y);
	else if (type == IQ4_NL)
		DecodedIQ4NL(block, y);
	else
		DecodedQ6K(block, y);
}

static uint64_t Elements(const Tensor* tensor)
{
	uint64_t elements = 1;
	for (uint32_t index = 0; index < tensor->dimensions; index++)
		elements *= tensor->shape[index];
	return elements;
}


/* Whether the file of the tensor holds all of its data, the file is open for reads then. */
static int Covered(const Tensor* tensor)
{
	if (!openings)
		openings = calloc(shardCount, sizeof(Opening)), openingSizes = calloc(shardCount, sizeof(uint64_t));
	if (!openings[tensor->shard])
		openings[tensor->shard] = OpenedFile(shardPaths[tensor->shard], &openingSizes[tensor->shard]);
	uint64_t end = tensor->start + tensor->offset + Elements(tensor) / BlockNumbers(tensor->type) * BlockBytes(tensor->type);
	return openings[tensor->shard] && end <= openingSizes[tensor->shard];
}

/* Reads the blocks of the tensor as they are stored, 0 and the failure name if it can't. */
static unsigned char* Blocks(const Tensor* tensor, uint64_t count, const char** failure)
{
	uint32_t type = tensor->type;
	uint64_t blockBytes = BlockBytes(type), blocks = count / BlockNumbers(type);
	if (blockBytes == 0 || count % BlockNumbers(type) != 0)
	{
		*failure = "OnlyF32F16BF16Q8_0Q5_1Q4_KQ5_KQ6_KAndIQ4_NLTensorsAreSupported";
		return 0;
	}
	unsigned char* raw = malloc(blocks * blockBytes + 1);
	if (Covered(tensor) && ReadFrom(openings[tensor->shard], tensor->start + tensor->offset, raw, blocks * blockBytes))
		return raw;
	free(raw);
	*failure = "GgufTensorDataTruncated";
	return 0;
}

/* Reads all numbers of the tensor as floats, 0 and the failure name if it can't. */
static float* Floats(const Tensor* tensor, uint64_t count, const char** failure)
{
	unsigned char* raw = Blocks(tensor, count, failure);
	float* floats = raw ? malloc(count * sizeof(float) + 1) : 0;
	uint64_t blockNumbers = BlockNumbers(tensor->type), blockBytes = BlockBytes(tensor->type);
	for (uint64_t block = 0; floats && block < count / blockNumbers; block++)
		Decoded(tensor->type, raw + block * blockBytes, floats + block * blockNumbers);
	free(raw);
	return floats;
}

static const Tensor* Requested(double path, double name, uint32_t dimensions, const char** failure)
{
	*failure = Loaded(strict_text(path));
	const Tensor* tensor = *failure ? 0 : Found(strict_text(name));
	if (!*failure && (!tensor || (dimensions && tensor->dimensions != dimensions)))
		*failure = "GgufTensorNotFound";
	return *failure ? 0 : tensor;
}

/* All numbers of the tensor name as doubles. */
STRICT_EXPORT double GgufFile_Values(double self, double path, double name)
{
	static strict_numbers* numbers = 0;
	const char* failure = 0;
	const Tensor* tensor = Requested(path, name, 0, &failure);
	uint64_t count = tensor ? Elements(tensor) : 0;
	float* floats = tensor ? Floats(tensor, count, &failure) : 0;
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
/* The blocks of one row of a Q4_K or Q6_K matrix as a packed row of 4 or 6 bits: Q4_K blocks stay as they are, Q6_K
   blocks give their half scale d to the end of the row. */
static void Repacked(const unsigned char* blocks, uint64_t bits, uint64_t width, strict_packed* packed)
{
	uint64_t count = width / 256;
	if (bits == 4)
		memcpy(packed->data, blocks, count * 144);
	for (uint64_t block = 0; bits == 6 && block < count; block++)
	{
		memcpy(packed->data + block * 208, blocks + block * 210, 208);
		memcpy(packed->data + count * 208 + block * 2, blocks + block * 210 + 208, 2);
	}
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
	unsigned char* raw = tensor && isBlocked ? Blocks(tensor, width * height, &failure) : 0;
	float* floats = tensor && !isBlocked ? Floats(tensor, width * height, &failure) : 0;
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

/* ggml's tables (ggml-cpu.c fills them in ggml_cpu_init) and the functions of ggml.c its quantization code refers to:
   the kernels used here only read the halves table, the rest is never used. */
float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

void ggml_abort(const char* file, int line, const char* format, ...)
{
	fprintf(stderr, "ggml failed at %s:%d: %s\n", file, line, format);
	abort();
}

size_t ggml_type_size(enum ggml_type type) { GGML_ABORT("ggml_type_size is not available"); }
size_t ggml_row_size(enum ggml_type type, int64_t count) { GGML_ABORT("ggml_row_size is not available"); }
const char* ggml_type_name(enum ggml_type type) { GGML_ABORT("ggml_type_name is not available"); }

typedef void (*Dot)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
typedef struct { Dot dot; void (*quantize)(const float*, void*, int64_t); uint64_t numbers; uint64_t bytes; } Product;

/* ggml's dot product of a row of type with an input quantized to the type ggml pairs with it, no dot without one. */
static Product Multiplied(uint32_t type)
{
	Product none = { 0, 0, 1, 0 };
	Product q4k = { ggml_vec_dot_q4_K_q8_K, quantize_row_q8_K, QK_K, sizeof(block_q8_K) };
	Product q5k = { ggml_vec_dot_q5_K_q8_K, quantize_row_q8_K, QK_K, sizeof(block_q8_K) };
	Product q51 = { ggml_vec_dot_q5_1_q8_1, quantize_row_q8_1, QK8_1, sizeof(block_q8_1) };
	Product q80 = { ggml_vec_dot_q8_0_q8_0, quantize_row_q8_0, QK8_0, sizeof(block_q8_0) };
	return type == Q4_K ? q4k : type == Q5_K ? q5k : type == Q5_1 ? q51 : type == Q8_0 ? q80 : none;
}

/* The stacked expert matrices name (columns, rows, experts) of the layer prefix, of a type ggml multiplies. */
static const Tensor* Stacked(double path, double prefix, const char* name, const char** failure)
{
	char full[256];
	snprintf(full, sizeof(full), "%s%s", strict_text(prefix), name);
	*failure = *failure ? *failure : Loaded(strict_text(path));
	const Tensor* tensor = *failure ? 0 : Found(full);
	if (!*failure && (!tensor || tensor->dimensions != 3))
		*failure = "GgufExpertsNotFound";
	else if (!*failure && (!Multiplied(tensor->type).dot || tensor->shape[0] % Multiplied(tensor->type).numbers != 0))
		*failure = "OnlyQ4_KQ5_KQ5_1AndQ8_0ExpertsAreSupported";
	else if (!*failure && !Covered(tensor))
		*failure = "GgufTensorDataTruncated";
	return *failure ? 0 : tensor;
}

static int IsIndex(double number, uint64_t count) { return number >= 0 && number < count && number == floor(number); }

static const char* Mismatched(const Tensor* gate, const Tensor* up, const Tensor* down, const strict_numbers* chosen,
	const strict_numbers* input)
{
	uint64_t hidden = gate->shape[0], width = gate->shape[1], count = gate->shape[2];
	if (memcmp(up->shape, gate->shape, sizeof(gate->shape)) != 0 || down->shape[0] != width ||
		down->shape[1] != hidden || down->shape[2] != count)
		return "GgufExpertsShapesDiffer";
	if (Multiplied(up->type).quantize != Multiplied(gate->type).quantize)
		return "GgufExpertsGateAndUpTypesDiffer";
	uint64_t tokens = (uint64_t)input->count / hidden;
	if (tokens == 0 || (uint64_t)input->count % hidden != 0 || (uint64_t)chosen->count % tokens != 0)
		return "ExpertsInputLengthDiffers";
	for (uint64_t index = 0; index < (uint64_t)chosen->count; index++)
		if (!IsIndex(chosen->items[index], count))
			return "ExpertNumberOutOfRange";
	return 0;
}

/* The values quantized to the type ggml multiplies rows of type with. */
static void* Quantized(uint32_t type, const float* values, uint64_t count)
{
	Product product = Multiplied(type);
	void* quantized = malloc(count / product.numbers * product.bytes + 1);
	product.quantize(values, quantized, (int64_t)count);
	return quantized;
}

#ifdef _WIN32
typedef SRWLOCK Lock;
typedef CONDITION_VARIABLE Condition;
static Lock poolLock = SRWLOCK_INIT;
static Condition poolStarted = CONDITION_VARIABLE_INIT, poolDone = CONDITION_VARIABLE_INIT;
static void Locked(Lock* lock) { AcquireSRWLockExclusive(lock); }
static void Unlocked(Lock* lock) { ReleaseSRWLockExclusive(lock); }
static void Waited(Condition* condition, Lock* lock) { SleepConditionVariableSRW(condition, lock, INFINITE, 0); }
static void Woken(Condition* condition) { WakeAllConditionVariable(condition); }
static uint64_t Processors(void) { return GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); }
#else
typedef pthread_mutex_t Lock;
typedef pthread_cond_t Condition;
static Lock poolLock = PTHREAD_MUTEX_INITIALIZER;
static Condition poolStarted = PTHREAD_COND_INITIALIZER, poolDone = PTHREAD_COND_INITIALIZER;
static void Locked(Lock* lock) { pthread_mutex_lock(lock); }
static void Unlocked(Lock* lock) { pthread_mutex_unlock(lock); }
static void Waited(Condition* condition, Lock* lock) { pthread_cond_wait(condition, lock); }
static void Woken(Condition* condition) { pthread_cond_broadcast(condition); }
static uint64_t Processors(void) { return (uint64_t)sysconf(_SC_NPROCESSORS_ONLN); }
#endif
#if defined(__x86_64__) || defined(_M_X64)
#define Paused() _mm_pause()
#else
#define Paused() ((void)0)
#endif
enum { Spins = 20000 };
static uint64_t poolThreads = 0, poolCount = 0;
static atomic_uint_fast64_t poolGeneration, poolBusy, poolNext;
static void (*poolWork)(void*, uint64_t) = 0;
static void* poolContext = 0;

static void Worked(void)
{
	for (uint64_t item = atomic_fetch_add(&poolNext, 1); item < poolCount; item = atomic_fetch_add(&poolNext, 1))
		poolWork(poolContext, item);
}

/* Waits until counter is no longer value, spinning a while (jobs follow each other quickly) before sleeping. */
static void Awaited(atomic_uint_fast64_t* counter, uint64_t value, Condition* condition)
{
	for (int spin = 0; atomic_load(counter) == value && spin < Spins; spin++)
		Paused();
	Locked(&poolLock);
	while (atomic_load(counter) == value)
		Waited(condition, &poolLock);
	Unlocked(&poolLock);
}

static void Wake(Condition* condition)
{
	Locked(&poolLock);
	Woken(condition);
	Unlocked(&poolLock);
}

/* A thread of the pool: works on every job started after it, the last one done wakes the starter. */
static void Serve(void)
{
	for (uint64_t seen = 0;; seen++)
	{
		Awaited(&poolGeneration, seen, &poolStarted);
		Worked();
		if (atomic_fetch_sub(&poolBusy, 1) == 1)
			Wake(&poolDone);
	}
}

#ifdef _WIN32
static DWORD WINAPI Served(void* unused) { Serve(); return 0; }
static int Started(void)
{
	HANDLE thread = CreateThread(0, 0, Served, 0, 0, 0);
	return thread && CloseHandle(thread);
}
#else
static void* Served(void* unused) { Serve(); return 0; }
static int Started(void)
{
	pthread_t thread;
	return pthread_create(&thread, 0, Served, 0) == 0 && pthread_detach(thread) == 0;
}
#endif

/* Runs work for the items 0 to count on this thread and the pool's: STRICT_THREADS in all, else a quarter of the
   logical processors (experts of Qwen3.8-Flash-Next on a Threadripper 3990X: 16 threads 47, 32 threads 40, 64 threads
   58 and 128 threads 135 ms per token). */
static void Parallel(void (*work)(void*, uint64_t), void* context, uint64_t count)
{
	const char* wanted = poolThreads ? 0 : getenv("STRICT_THREADS");
	uint64_t threads = poolThreads ? 0 : wanted && atoi(wanted) > 0 ? (uint64_t)atoi(wanted) : Processors() / 4;
	for (poolThreads = poolThreads ? poolThreads : 1; poolThreads < threads && Started(); )
		poolThreads++;
	poolWork = work;
	poolContext = context;
	poolCount = count;
	atomic_store(&poolNext, 0);
	atomic_store(&poolBusy, poolThreads - 1);
	atomic_fetch_add(&poolGeneration, 1);
	Wake(&poolStarted);
	Worked();
	for (uint64_t busy = atomic_load(&poolBusy); busy > 0; busy = atomic_load(&poolBusy))
		Awaited(&poolBusy, busy, &poolDone);
}

enum { RowChunk = 16 };
typedef struct
{
	unsigned char* rows;
	uint64_t rowBytes, width, height, offset;
	Opening file;
	Dot dot;
	const void* input;
	float* output;
} Task;
typedef struct { const Task* tasks; uint64_t chunks; } Job;

/* The rows of the matrix of expert in stacked (read by Staged) times the quantized input into output. */
static Task Rows(const Tensor* stacked, uint64_t expert, const void* input, float* output)
{
	uint64_t width = stacked->shape[0], height = stacked->shape[1];
	uint64_t rowBytes = width / BlockNumbers(stacked->type) * BlockBytes(stacked->type);
	uint64_t offset = stacked->start + stacked->offset + expert * height * rowBytes;
	Task task = { 0, rowBytes, width, height, offset, openings[stacked->shard], Multiplied(stacked->type).dot, input, output };
	return task;
}

static atomic_int stagingFailed;

static void Fetched(void* context, uint64_t item)
{
	Task* task = &((Task*)context)[item];
	if (!ReadFrom(task->file, task->offset, task->rows, task->height * task->rowBytes))
		atomic_store(&stagingFailed, 1);
}

/* Reads the rows of all tasks into one buffer kept for the next call, one read per matrix on all threads: reading
   them from the file cache is copying, the pages of a mapping fault one by one (40 to 120 thousand a second). */
static int Staged(Task* tasks, uint64_t count)
{
	static unsigned char* staged = 0;
	static uint64_t stagedBytes = 0;
	uint64_t total = 0;
	for (uint64_t index = 0; index < count; index++)
		total += tasks[index].height * tasks[index].rowBytes;
	if (total > stagedBytes)
		free(staged), staged = malloc(total), stagedBytes = total;
	for (uint64_t index = 0, at = 0; index < count; at += tasks[index].height * tasks[index].rowBytes, index++)
		tasks[index].rows = staged + at;
	atomic_store(&stagingFailed, 0);
	Parallel(Fetched, tasks, count);
	return !atomic_load(&stagingFailed);
}

/* RowChunk rows of a task of the job, one ggml dot product per row. */
static void Products(void* context, uint64_t item)
{
	const Job* job = context;
	const Task* task = &job->tasks[item / job->chunks];
	uint64_t first = item % job->chunks * RowChunk, end = first + RowChunk;
	for (uint64_t row = first; row < end && row < task->height; row++)
		task->dot((int)task->width, task->output + row, 0, task->rows + row * task->rowBytes, 0, task->input, 0, 1);
}

/* The tasks (all of the same height) on all threads. */
static void Ran(const Task* tasks, uint64_t count)
{
	Job job = { tasks, (tasks[0].height + RowChunk - 1) / RowChunk };
	Parallel(Products, &job, count * job.chunks);
}

/* silu(gate) * up of every expert quantized for the down matrices. */
static unsigned char* Activated(const Tensor* down, float* gated, const float* upped, uint64_t count)
{
	uint64_t width = down->shape[0], bytes = width / Multiplied(down->type).numbers * Multiplied(down->type).bytes;
	unsigned char* activated = malloc(count * bytes + 1);
	for (uint64_t column = 0; column < count * width; column++)
		gated[column] = gated[column] / (1 + expf(-gated[column])) * upped[column];
	for (uint64_t index = 0; index < count; index++)
		Multiplied(down->type).quantize(gated + index * width, activated + index * bytes, (int64_t)width);
	return activated;
}

/* Stages the gate, up and down rows of each distinct expert of the tasks once (pairs of gate and up tasks, then the
   downs) and points the tasks of every token choosing it at them. */
static int Shared(Task* tasks, uint64_t count, const strict_numbers* experts, uint64_t expertCount)
{
	uint64_t* slots = malloc((expertCount + count + 1) * sizeof(uint64_t)), * chosen = slots + expertCount, distinct = 0;
	Task* staging = malloc((count * 3 + 1) * sizeof(Task));
	for (uint64_t expert = 0; expert < expertCount; expert++)
		slots[expert] = count;
	for (uint64_t index = 0; index < count; index++)
	{
		uint64_t expert = (uint64_t)experts->items[index];
		if (slots[expert] == count)
		{
			staging[distinct * 3] = tasks[index * 2], staging[distinct * 3 + 1] = tasks[index * 2 + 1];
			staging[distinct * 3 + 2] = tasks[count * 2 + index], slots[expert] = distinct++;
		}
		chosen[index] = slots[expert];
	}
	int isStaged = Staged(staging, distinct * 3);
	for (uint64_t index = 0; index < count; index++)
	{
		tasks[index * 2].rows = staging[chosen[index] * 3].rows;
		tasks[index * 2 + 1].rows = staging[chosen[index] * 3 + 1].rows;
		tasks[count * 2 + index].rows = staging[chosen[index] * 3 + 2].rows;
	}
	free(staging), free(slots);
	return isStaged;
}

/* The outputs of the chosen experts of the layer prefix for each input token (as many chosen experts per token) one
   after the other like llama.cpp's CPU backend computes them: down(silu(gate input) * up input), each product ggml's
   dot product of rows as stored (read from the files, Q4_K, Q5_K, Q5_1 or Q8_0) with the input quantized for them, the
   rows spread over all threads, an expert chosen by several tokens read once. */
STRICT_EXPORT double GgufFile_ExpertOutputs(double self, double path, double prefix, double chosen, double input)
{
	static strict_numbers* outputs = 0;
	static int isTabled = 0;
	const char* failure = 0;
	const Tensor* gate = Stacked(path, prefix, "ffn_gate_exps.weight", &failure);
	const Tensor* up = Stacked(path, prefix, "ffn_up_exps.weight", &failure);
	const Tensor* down = Stacked(path, prefix, "ffn_down_exps.weight", &failure);
	const strict_numbers* experts = strict_numbers_of(chosen), * values = strict_numbers_of(input);
	failure = failure ? failure : Mismatched(gate, up, down, experts, values);
	if (failure)
		return strict_fail(failure);
	for (uint32_t bits = 0; !isTabled && bits < 65536; bits++)
		ggml_table_f32_f16[bits] = ggml_compute_fp16_to_fp32((ggml_fp16_t)bits);
	isTabled = 1;
	uint64_t hidden = gate->shape[0], width = gate->shape[1], count = (uint64_t)experts->count;
	uint64_t tokens = (uint64_t)values->count / hidden, used = count / tokens;
	float* floats = malloc((tokens * hidden + count * (width * 2 + hidden) + 1) * sizeof(float));
	float* gated = floats + tokens * hidden, * upped = gated + count * width, * downs = upped + count * width;
	for (uint64_t index = 0; index < tokens * hidden; index++)
		floats[index] = (float)values->items[index];
	unsigned char* quantized = Quantized(gate->type, floats, tokens * hidden);
	uint64_t inputBytes = hidden / Multiplied(gate->type).numbers * Multiplied(gate->type).bytes;
	Task* tasks = malloc((count * 3 + 1) * sizeof(Task));
	for (uint64_t index = 0; index < count; index++)
	{
		uint64_t expert = (uint64_t)experts->items[index];
		const unsigned char* token = quantized + index / used * inputBytes;
		tasks[index * 2] = Rows(gate, expert, token, gated + index * width);
		tasks[index * 2 + 1] = Rows(up, expert, token, upped + index * width);
		tasks[count * 2 + index] = Rows(down, expert, 0, downs + index * hidden);
	}
	if (!Shared(tasks, count, experts, gate->shape[2]))
	{
		free(tasks), free(quantized), free(floats);
		return strict_fail("GgufTensorDataTruncated");
	}
	Ran(tasks, count * 2);
	unsigned char* activated = Activated(down, gated, upped, count);
	uint64_t activatedBytes = width / Multiplied(down->type).numbers * Multiplied(down->type).bytes;
	for (uint64_t index = 0; index < count; index++)
		tasks[count * 2 + index].input = activated + index * activatedBytes;
	Ran(tasks + count * 2, count);
	free(outputs);
	outputs = malloc(sizeof(strict_numbers) + count * hidden * sizeof(double));
	outputs->count = (double)(count * hidden);
	for (uint64_t index = 0; index < count * hidden; index++)
		outputs->items[index] = downs[index];
	free(activated);
	free(tasks);
	free(quantized);
	free(floats);
	return strict_numbers_result(outputs);
}

/* The numbers of the rows (the contiguous dimension) of the tensor name one after the other, decoded like Values, read
   row by row, so a huge table (the n-gram embeddings of Qwen3.8-Flash-Next) is never loaded as a whole. */
STRICT_EXPORT double GgufFile_Picked(double self, double path, double name, double rows)
{
	static strict_numbers* picked = 0;
	const char* failure = 0;
	const Tensor* tensor = Requested(path, name, 0, &failure);
	const strict_numbers* chosen = strict_numbers_of(rows);
	uint32_t type = tensor ? tensor->type : F32;
	uint64_t width = tensor ? tensor->shape[0] : 0, height = tensor ? Elements(tensor) / width : 0;
	if (tensor && (BlockBytes(type) == 0 || width % BlockNumbers(type) != 0))
		failure = "OnlyF32F16BF16Q8_0Q5_1Q4_KQ5_KQ6_KAndIQ4_NLTensorsAreSupported";
	failure = failure ? failure : Covered(tensor) ? 0 : "GgufTensorDataTruncated";
	for (uint64_t index = 0; !failure && index < (uint64_t)chosen->count; index++)
		if (!IsIndex(chosen->items[index], height))
			failure = "GgufRowOutOfRange";
	if (failure)
		return strict_fail(failure);
	uint64_t count = (uint64_t)chosen->count, blockNumbers = BlockNumbers(type);
	uint64_t rowBytes = width / blockNumbers * BlockBytes(type);
	float decoded[256];
	unsigned char* row = malloc(rowBytes + 1);
	free(picked);
	picked = malloc(sizeof(strict_numbers) + count * width * sizeof(double));
	picked->count = (double)(count * width);
	for (uint64_t index = 0; index < count; index++)
	{
		uint64_t offset = tensor->start + tensor->offset + (uint64_t)chosen->items[index] * rowBytes;
		if (!ReadFrom(openings[tensor->shard], offset, row, rowBytes))
			memset(row, 0, rowBytes), failure = "GgufTensorDataTruncated";
		for (uint64_t column = 0; column < width; column += blockNumbers)
		{
			Decoded(type, row + column / blockNumbers * BlockBytes(type), decoded);
			for (uint64_t number = 0; number < blockNumbers; number++)
				picked->items[index * width + column + number] = decoded[number];
		}
	}
	free(row);
	return failure ? strict_fail(failure) : strict_numbers_result(picked);
}
