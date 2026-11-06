#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include "strict.h"
/* Reading and converting rows is throughput work, its pool (one per file) takes every logical processor: on 8 threads of
   an i9-13900KF the NVFP4 rows of the H3 text encoder waited for unpacking and quantizing. */
#define PARALLEL_SHARE 1
#include "Parallel.h"
#include "ThreadState.h"
static _Thread_local char* entry;
static _Thread_local unsigned char* floatStorage;
static _Thread_local unsigned char* chunkBuffer;
static _Thread_local uint64_t chunkBufferCap;
static _Thread_local float* chunkFloats;
static _Thread_local uint64_t chunkFloatsCap;
/* PROFILE (temporary) */
static double ProfileNow(void)
{
#ifdef _WIN32
	LARGE_INTEGER count, frequency;
	QueryPerformanceCounter(&count);
	QueryPerformanceFrequency(&frequency);
	return (double)count.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
	return 0;
#endif
}
static int ProfileOn(void)
{
	static int on = -1;
	if (on < 0)
		on = getenv("STRICT_LOAD_LOG") != 0;
	return on;
}
static double profileStart, profileRows, profilePlaced, profileRead;
static void Profiled(const char* what, uint64_t height, uint64_t width, double started)
{
	if (!ProfileOn())
		return;
	double now = ProfileNow();
	if (!profileStart)
		profileStart = started;
	printf("load %s %llux%llu at %.1f took %.2f rows %.2f placed %.2f read %.2f\n", what, (unsigned long long)height,
		(unsigned long long)width, started - profileStart, now - started, profileRows, profilePlaced, profileRead);
	profileRows = profilePlaced = profileRead = 0;
}
#ifndef _WIN32
#include <sys/stat.h>
#endif
/* The headers of the files read last, each kept while its file has the same size and modification time: every tensor
   lookup read the whole header again (60-130 KB for a LoRA of Moss, about 5 lookups per merged linear). */
enum { HeaderCount = 8 };
typedef struct { char* path; uint64_t size, modified, length; char* text; } Header;
static _Thread_local Header headers[HeaderCount];
static _Thread_local int nextHeader;

#ifdef __EMSCRIPTEN__
/* A file that is not in the package of a web build is read from the server the page came from, with range requests
   (WebRuntime.js), 8 bit rows go straight to WebGPU (strict_rows_web). */
int strict_webfile_read(const char* url, double offset, double bytes, void* target);
int strict_webfile_rows(const char* url, double offset, double rows, double rowBytes, double codesBytes, double stride,
	double columns, const float* scales);
double strict_rows_web(uint64_t count, uint64_t elements, double bits, int64_t handle, int64_t nibbles);
static int IsRemote(const char* path)
{
	FILE* file = fopen(path, "rb");
	if (file)
		fclose(file);
	return !file;
}
#else
static int IsRemote(const char* path) { return 0; }
static int strict_webfile_read(const char* url, double offset, double bytes, void* target) { return 0; }
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

/* The size and modification time of the file at path, 0 when it can't be read. */
static int Stamped(const char* path, uint64_t* size, uint64_t* modified)
{
#ifdef _WIN32
	WIN32_FILE_ATTRIBUTE_DATA data;
	if (!GetFileAttributesExA(path, GetFileExInfoStandard, &data))
		return 0;
	*size = ((uint64_t)data.nFileSizeHigh << 32) | data.nFileSizeLow;
	*modified = ((uint64_t)data.ftLastWriteTime.dwHighDateTime << 32) | data.ftLastWriteTime.dwLowDateTime;
#else
	struct stat status;
	if (stat(path, &status))
		return 0;
	*size = (uint64_t)status.st_size;
	*modified = (uint64_t)status.st_mtime;
#endif
	return 1;
}

/* Removes the whitespace outside of strings from the JSON text of length (writers like Python's json.dumps put spaces
   after , and :), all lookups then find "name":{ and "shape":[2,3]. */
static void Compacted(char* text, uint64_t length)
{
	uint64_t kept = 0;
	int isString = 0, isEscaped = 0;
	for (uint64_t index = 0; index < length; index++)
	{
		char character = text[index];
		if (!isString && (character == ' ' || character == '\t' || character == '\r' || character == '\n'))
			continue;
		if (isEscaped)
			isEscaped = 0;
		else if (isString && character == '\\')
			isEscaped = 1;
		else if (character == '"')
			isString = !isString;
		text[kept++] = character;
	}
	text[kept] = 0;
}

/* The JSON header of the .safetensors file at path from headers or read into them, 0 when it can't be read. */
/* The JSON header of a file read from the server of the page (IsRemote), length bytes, 0 when it can't be read. */
static char* RemoteHeader(const char* path, uint64_t* length)
{
	unsigned char size[8];
	if (!strict_webfile_read(path, 0, 8, size))
		return 0;
	*length = 0;
	for (int index = 7; index >= 0; index--)
		*length = *length * 256 + size[index];
	char* text = *length <= 100000000 ? malloc(*length + 1) : 0;
	if (text && !strict_webfile_read(path, 8, (double)*length, text))
	{
		free(text);
		return 0;
	}
	return text;
}

static const char* HeaderOf(const char* path, const char** failure)
{
	uint64_t size = 0, modified = 0, length = 0;
	*failure = "TensorFileNotReadable";
	int isRemote = IsRemote(path);
	if (!isRemote && !Stamped(path, &size, &modified))
		return 0;
	for (int index = 0; index < HeaderCount; index++)
		if (headers[index].path && !strcmp(headers[index].path, path) && headers[index].size == size &&
			headers[index].modified == modified)
			return headers[index].text;
	char* text = 0;
	size_t read = 0;
	if (isRemote)
	{
		text = RemoteHeader(path, &length);
		read = text ? length : 0;
	}
	else
	{
		FILE* file = Opened(path, &length);
		if (!file || length > 100000000)
		{
			if (file)
				fclose(file);
			return 0;
		}
		text = malloc(length + 1);
		read = text ? fread(text, 1, length, file) : 0;
		fclose(file);
	}
	if (!text || read != length)
	{
		free(text);
		*failure = "TensorFileHeaderTruncated";
		return 0;
	}
	Compacted(text, length);
	Header* slot = &headers[nextHeader];
	nextHeader = (nextHeader + 1) % HeaderCount;
	free(slot->path);
	free(slot->text);
	size_t pathLength = strlen(path) + 1;
	slot->path = malloc(pathLength);
	memcpy(slot->path, path, pathLength);
	slot->size = size;
	slot->modified = modified;
	slot->length = length;
	slot->text = text;
	return text;
}

/* Where the tensor data of the file at path starts (after its header), 0 when it can't be read. */
static uint64_t DataStart(const char* path)
{
	const char* failure = 0;
	const char* text = HeaderOf(path, &failure);
	for (int index = 0; text && index < HeaderCount; index++)
		if (headers[index].text == text)
			return 8 + headers[index].length;
	return 0;
}

STRICT_EXPORT double TensorFile_Header(double self, double path)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	const char* failure = 0;
	const char* text = HeaderOf(strict_text(path), &failure);
	return text ? strict_text_result(text) : strict_fail(failure);
}

/* Whether the header's _quantization_metadata names the layer of the weight name as ComfyUI's convrot_w4a4: its I8
   rows hold two signed 4 bit numbers per byte, the lower nibble first. */
static int IsNibbled(const char* header, const char* name)
{
	if (!strstr(header, "convrot_w4a4"))
		return 0;
	size_t length = strlen(name);
	char stackKey[256];
	char* key = length > 7 && !strcmp(name + length - 7, ".weight") ?
		(length + 40 <= sizeof(stackKey) ? stackKey : malloc(length + 40)) : 0;
	if (!key)
		return 0;
	snprintf(key, length + 40, "\\\"%.*s\\\": {\\\"format\\\": \\\"convrot_w4a4\\\"", (int)(length - 7), name);
	int isFound = strstr(header, key) != 0;
	if (key != stackKey)
		free(key);
	return isFound;
}

/* Rewrites the entry of an I8 matrix of nibbles (with room for a few more characters) as I4 with all its columns. */
static void Nibbled(char* text)
{
	unsigned long long height = 0, width = 0, first = 0, last = 0;
	const char* shape = strstr(text, "\"shape\":[");
	const char* offsets = strstr(text, "\"data_offsets\":[");
	if (strstr(text, "\"dtype\":\"I8\"") && shape && offsets && sscanf(shape, "\"shape\":[%llu,%llu]", &height, &width) == 2 &&
		sscanf(offsets, "\"data_offsets\":[%llu,%llu]", &first, &last) == 2)
		sprintf(text, "\"dtype\":\"I4\",\"shape\":[%llu,%llu],\"data_offsets\":[%llu,%llu]", height, 2 * width, first, last);
}

/* The entry of the tensor name in the header of the file at path ("dtype":"F32","shape":[3],"data_offsets":[0,12]),
   empty when the header has none; convrot_w4a4 weights are I4 matrices (IsNibbled). */
STRICT_EXPORT double TensorFile_Entry(double self, double path, double name)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	const char* failure = 0;
	const char* text = HeaderOf(strict_text(path), &failure);
	if (!text)
		return strict_fail(failure);
	const char* wanted = strict_text(name);
	size_t nameLength = strlen(wanted);
	char stackKey[256];
	char* key = nameLength + 5 <= sizeof(stackKey) ? stackKey : malloc(nameLength + 5);
	if (!key)
		return strict_fail("OutOfMemory");
	key[0] = '"';
	memcpy(key + 1, wanted, nameLength);
	memcpy(key + 1 + nameLength, "\":{", 4);
	const char* found = strstr(text, key);
	if (key != stackKey)
		free(key);
	const char* open = found ? found + nameLength + 4 : "";
	const char* close = strchr(open, '}');
	size_t entryLength = close ? (size_t)(close - open) : strlen(open);
	free(entry);
	entry = malloc(entryLength + 64);
	memcpy(entry, open, entryLength);
	entry[entryLength] = 0;
	if (IsNibbled(text, wanted))
		Nibbled(entry);
	return strict_text_result(entry);
}

/* A tensor as its header entry describes it: "dtype":"F16","shape":[2,3],"data_offsets":[24,36]. I4 rows hold two
   numbers per byte (isNibbles, width counts the bytes). */
typedef struct { uint64_t bytes, isBrain, isSigned, isEight, isNibbles, rank, height, width, start; } Tensor;

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
	tensor->isNibbles = strncmp(type, "I4\"", 3) == 0;
	tensor->isSigned = strncmp(type, "I8\"", 3) == 0 || tensor->isNibbles;
	tensor->isEight = strncmp(type, "F8_E4M3\"", 8) == 0;
	tensor->bytes = strncmp(type, "F32\"", 4) == 0 ? 4 : strncmp(type, "F16\"", 4) == 0 || tensor->isBrain ? 2 :
		tensor->isSigned || tensor->isEight || strncmp(type, "U8\"", 3) == 0 ? 1 : 0;
	if (!tensor->bytes)
		return "OnlyF32F16BF16F8I8AndU8TensorsAreSupported";
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
	if (tensor->isNibbles && (tensor->rank != 2 || tensor->width % 2))
		return "TensorShapeMalformed";
	tensor->width /= tensor->isNibbles ? 2 : 1;
	tensor->start = range[0];
	if (range[1] < range[0] || range[1] - range[0] != tensor->height * tensor->width * tensor->bytes)
		return "TensorSizeMismatch";
	return 0;
}

/* The 8 bit floats of NVFP4 block scales, Eight reads them: filled by EightTableInit before any use */
static float EightTable[256];
static int EightTableReady = 0;
static void EightTableInit(void)
{
	if (EightTableReady)
		return;
	for (int i = 0; i < 256; i++)
	{
		unsigned char bits = (unsigned char)i;
		uint32_t exponent = (bits >> 3) & 15, mantissa = bits & 7;
		union { uint32_t bits; float number; } slot = { (uint32_t)(bits & 128) << 24 | (exponent + 120) << 23 | mantissa << 20 };
		EightTable[i] = exponent ? slot.number : (bits & 128 ? -1.0f : 1.0f) * (float)mantissa * 0.001953125f;
	}
	EightTableReady = 1;
}

static inline float Eight(unsigned char bits)
{
	return EightTable[bits];
}

/* Widens count numbers in the format of tensor at source to floats at target, which may start at or before source as
   long as the numbers end together (every number is read before a float overwrites it). */
static void Widened(const unsigned char* source, const Tensor* tensor, float* target, uint64_t count)
{
	if (tensor->bytes == 4)
	{
		memmove(target, source, count * sizeof(float));
		return;
	}
#if defined(__AVX2__)
	if (tensor->bytes == 2 && tensor->isBrain)
	{
		const uint16_t* src16 = (const uint16_t*)source;
		uint64_t index = 0;
		for (; index + 16 <= count; index += 16)
		{
			__m128i in0 = _mm_loadu_si128((const __m128i*)(src16 + index));
			__m128i in1 = _mm_loadu_si128((const __m128i*)(src16 + index + 8));
			__m256i s0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(in0), 16);
			__m256i s1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(in1), 16);
			_mm256_storeu_si256((__m256i*)(target + index), s0);
			_mm256_storeu_si256((__m256i*)(target + index + 8), s1);
		}
		for (; index < count; index++)
		{
			union { uint32_t bits; float val; } u = { (uint32_t)src16[index] << 16 };
			target[index] = u.val;
		}
		return;
	}
#endif
#if defined(__F16C__) || defined(__AVX2__)
	if (tensor->bytes == 2 && !tensor->isBrain)
	{
		uint64_t index = 0;
		for (; index + 8 <= count; index += 8)
		{
			__m128i in = _mm_loadu_si128((const __m128i*)(source + 2 * index));
			__m256 f = _mm256_cvtph_ps(in);
			_mm256_storeu_ps(target + index, f);
		}
		for (; index < count; index++)
			target[index] = strict_half_float((uint16_t)(source[2 * index] | source[2 * index + 1] << 8));
		return;
	}
#endif
	/* a condition in the loop header keeps clang from vectorizing (10 times slower), so each loop has its own if */
	if (tensor->bytes == 2 && tensor->isBrain)
		for (uint64_t index = 0; index < count; index++)
			target[index] = strict_brain_float((uint16_t)(source[2 * index] | source[2 * index + 1] << 8));
	else if (tensor->bytes == 2)
		for (uint64_t index = 0; index < count; index++)
			target[index] = strict_half_float((uint16_t)(source[2 * index] | source[2 * index + 1] << 8));
	else if (tensor->bytes == 1)
	{
		EightTableInit();
		if (tensor->isEight)
			for (uint64_t index = 0; index < count; index++)
				target[index] = EightTable[source[index]];
		else if (tensor->isSigned)
			for (uint64_t index = 0; index < count; index++)
				target[index] = (float)(int8_t)source[index];
		else
			for (uint64_t index = 0; index < count; index++)
				target[index] = (float)source[index];
	}
}

/* Reads the numbers of a tensor as they are in the file into a new block with room for widened bytes per number (the
   numbers at its end), 0 and the failure if it can't. */
static unsigned char* Loaded(double path, const Tensor* tensor, uint64_t widened, const char** failure)
{
	uint64_t length = 0, count = tensor->height * tensor->width;
	if (tensor->isNibbles)
	{
		*failure = "I4TensorsAreOnlyReadAsQuantizedRows";
		return 0;
	}
	int isRemote = IsRemote(strict_text(path));
	FILE* file = isRemote ? 0 : Opened(strict_text(path), &length);
	uint64_t start = isRemote ? DataStart(strict_text(path)) : 8 + length;
	if (!file && !start)
	{
		*failure = "TensorFileNotReadable";
		return 0;
	}
	unsigned char* block = malloc(count * widened + 1);
	if (!block)
	{
		if (file)
			fclose(file);
		*failure = "TensorTooLarge";
		return 0;
	}
	unsigned char* numbers = block + count * (widened - tensor->bytes);
	size_t read = 0;
	if (file)
	{
		fclose(file);
		uint64_t size = 0;
		Opening opening = OpenedFile(strict_text(path), &size);
		read = opening && ReadParts(opening, start + tensor->start, numbers, count * tensor->bytes) ? count : 0;
		if (opening)
			Closed(opening);
	}
	else
		read = strict_webfile_read(strict_text(path), (double)(start + tensor->start), (double)(count * tensor->bytes),
			numbers) ? count : 0;
	if (read != count)
	{
		free(block);
		*failure = "TensorFileDataTruncated";
		return 0;
	}
	return block;
}

/* Reads the numbers of a tensor as 32 bit floats, 0 and the failure if it can't. */
static float* Read(double path, const Tensor* tensor, const char** failure)
{
	float* floats = (float*)Loaded(path, tensor, sizeof(float), failure);
	if (floats)
		Widened((unsigned char*)floats + tensor->height * tensor->width * (sizeof(float) - tensor->bytes), tensor, floats,
			tensor->height * tensor->width);
	return floats;
}

static void ThreadReleased(void)
{
	for (int index = 0; index < HeaderCount; index++)
	{
		free(headers[index].path);
		free(headers[index].text);
	}
	free(entry);
	free(floatStorage);
	free(chunkBuffer);
	chunkBuffer = 0;
	chunkBufferCap = 0;
	free(chunkFloats);
	chunkFloats = 0;
	chunkFloatsCap = 0;
}

/* Packs count floats (stride apart) into the numbers at data with bits 32, 17 (rounded to bfloat16), 16 (rounded to
   halves) or 8 (strict_quantized). */
static void Packed(unsigned char* data, const float* first, uint64_t count, uint64_t stride, double bits)
{
	if (bits == 8)
		strict_quantized(data, first, count, stride);
	else if (bits == 32 && stride == 1)
		memcpy(data, first, count * sizeof(float));
	else if (bits == 16)
		strict_halves((uint16_t*)data, first, count, stride);
	else if (bits == 17)
		for (uint64_t column = 0; column < count; column++)
			((uint16_t*)data)[column] = strict_brain(first[column * stride]);
	else
		for (uint64_t column = 0; column < count; column++)
			((float*)data)[column] = first[column * stride];
}

/* NVFP4 rows: two 4 bit e2m1 numbers per byte (the first in the high nibble), each 16 scaled by its 8 bit float in the
   swizzled cuBLAS layout of blocks (tiles of 128 rows by 4 scales), all by scale and each column by its input factor. */
typedef struct { const unsigned char* blocks; uint64_t blockWidth; float scale; const float* inputs; int isCompact;
	const float* factors; } Fours;
static const float Nibbles[16] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6 };

/* The block scales stay the bytes of the file, each widened once for its 16 numbers. */
static void Unpacked(const Fours* fours, uint64_t row, const unsigned char* source, float* target, uint64_t bytes)
{
	const unsigned char* rowBlocks = fours->blocks + (row / 128) * (fours->blockWidth / 4) * 512 + (row % 32) * 16 +
		(row % 128 / 32) * 4;
	const float* inputs = fours->inputs;
	float rowScale = fours->scale * (fours->factors ? fours->factors[row] : 1);
	uint64_t blocks = bytes / 8, block = 0;
#ifdef __AVX2__
	const __m256 magnitudes = _mm256_setr_ps(0, 0.5f, 1, 1.5f, 2, 3, 4, 6);
	for (; block < blocks; block++)
	{
		__m256 scale = _mm256_set1_ps(Eight(rowBlocks[(block / 4) * 512 + (block & 3)]) * rowScale);
		__m128i packed = _mm_loadl_epi64((const __m128i*)(source + block * 8)), nibble = _mm_set1_epi8(15);
		__m128i codes = _mm_unpacklo_epi8(_mm_and_si128(_mm_srli_epi16(packed, 4), nibble), _mm_and_si128(packed, nibble));
		for (int half = 0; half < 2; half++)
		{
			__m256i code = _mm256_cvtepu8_epi32(half ? _mm_srli_si128(codes, 8) : codes);
			__m256 value = _mm256_mul_ps(_mm256_xor_ps(_mm256_permutevar8x32_ps(magnitudes, code),
				_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(code, 3), 31))), scale);
			if (inputs)
				value = _mm256_mul_ps(value, _mm256_loadu_ps(inputs + block * 16 + half * 8));
			_mm256_storeu_ps(target + block * 16 + half * 8, value);
		}
	}
#endif
	for (; block < blocks; block++)
	{
		float scale = Eight(rowBlocks[(block / 4) * 512 + (block & 3)]) * rowScale;
		const unsigned char* blockSource = source + block * 8;
		float* blockTarget = target + block * 16;
		if (inputs)
		{
			const float* blockInputs = inputs + block * 16;
			for (uint64_t byteIndex = 0; byteIndex < 8; byteIndex++)
			{
				unsigned char packed = blockSource[byteIndex];
				blockTarget[byteIndex * 2] = Nibbles[packed >> 4] * scale * blockInputs[byteIndex * 2];
				blockTarget[byteIndex * 2 + 1] = Nibbles[packed & 15] * scale * blockInputs[byteIndex * 2 + 1];
			}
		}
		else
			for (uint64_t byteIndex = 0; byteIndex < 8; byteIndex++)
			{
				unsigned char packed = blockSource[byteIndex];
				blockTarget[byteIndex * 2] = Nibbles[packed >> 4] * scale;
				blockTarget[byteIndex * 2 + 1] = Nibbles[packed & 15] * scale;
			}
	}
	for (uint64_t column = blocks * 16; column < 2 * bytes; column++)
	{
		uint64_t block = column / 16;
		float scale = Eight(rowBlocks[(block / 4) * 512 + (block & 3)]) * rowScale;
		unsigned char nibble = column % 2 ? source[column / 2] & 15 : source[column / 2] >> 4;
		target[column] = Nibbles[nibble] * scale * (inputs ? inputs[column] : 1);
	}
}

/* An NVFP4 row as a row of format 5 (strict.h, kept on the GPU as it is, strict_gpu_expanded): its codes as they are,
   the 8 bit float scales of its blocks out of their swizzled tiles, then the factor of the row (the scale of the matrix,
   times the factor of the row when given). */
static void Fourths(const Fours* fours, uint64_t row, const unsigned char* source, unsigned char* target, uint64_t bytes)
{
	const unsigned char* rowBlocks = fours->blocks + (row / 128) * (fours->blockWidth / 4) * 512 + (row % 32) * 16 +
		(row % 128 / 32) * 4;
	uint64_t groups = (2 * bytes + 31) / 32, blocks = bytes / 8, factorAt = (groups * 18 + 3) / 4 * 4;
	memmove(target, source, bytes);
	memset(target + bytes, 0, groups * 16 - bytes);
	unsigned char* scales = target + groups * 16;
	for (uint64_t block = 0; block < blocks; block++)
		scales[block] = rowBlocks[(block / 4) * 512 + (block & 3)];
	memset(scales + blocks, 0, factorAt - groups * 16 - blocks);
	float factor = fours->scale * (fours->factors ? fours->factors[row] : 1);
	memcpy(target + factorAt, &factor, sizeof(float));
}

enum { ChunkRows = 64 };
typedef struct { Opening file; const Tensor* tensor; uint64_t start; unsigned char* data; uint64_t stride; double bits;
	int isStored; const float* scales; const Fours* fours; atomic_int failed; const unsigned char* read; uint64_t first;
	uint64_t last; } Placing;

/* A stored row read to source moved to its place (they may overlap), as a row of format 5 for NVFP4 rows, 8 bit rows
   with a scale each get it for all their 32 numbers. */
static void Spread(const Placing* placing, uint64_t row, const unsigned char* source, uint64_t columns)
{
	unsigned char* target = placing->data + row * placing->stride;
	if (placing->fours)
	{
		Fourths(placing->fours, row, source, target, placing->tensor->width);
		return;
	}
	memmove(target, source, placing->tensor->width * placing->tensor->bytes);
	float* parts = (float*)(target + strict_packed_data(columns, 8));
	for (uint64_t part = 0; placing->scales && part < (columns + 31) / 32; part++)
		parts[part] = placing->scales[row];
}

/* Reads the rows of one chunk at once into a buffer and copies them to their place, converted unless stored, 8 bit rows
   with a scale each get it for all their 32 numbers. Stored rows are read straight to the place of the chunk and spread
   to their stride from the last row on (a row only moves up over rows read after it, which moved already): through the
   buffer they were copied once more (13 GB of NVFP4 rows of the H3 text encoder, 21 GB of 8 bit rows of H3). */
static void PlacedChunk(void* context, uint64_t chunk)
{
	Placing* placing = context;
	const Tensor* tensor = placing->tensor;
	uint64_t rowBytes = tensor->width * tensor->bytes, first = placing->first + chunk * ChunkRows;
	uint64_t end = placing->last ? placing->last : tensor->height;
	uint64_t count = end - first < ChunkRows ? end - first : ChunkRows;
	uint64_t columns = placing->fours || tensor->isNibbles ? 2 * tensor->width : tensor->width;
#ifndef STRICT_LIBRARY
	/* only native rows have nothing but padding between them, those of the VM library (strict.h) a header each */
	if (!placing->read && placing->isStored && !tensor->isNibbles && placing->stride >= rowBytes &&
		(!placing->fours || placing->fours->isCompact))
	{
		unsigned char* region = placing->data + first * placing->stride;
		if (!ReadFrom(placing->file, placing->start + first * rowBytes, region, count * rowBytes))
		{
			atomic_store(&placing->failed, 1);
			return;
		}
		for (uint64_t back = count; back > 0; back--)
			Spread(placing, first + back - 1, region + (back - 1) * rowBytes, columns);
		return;
	}
#endif
	uint64_t bufferNeeded = count * rowBytes + 1;
	if (bufferNeeded > chunkBufferCap)
	{
		free(chunkBuffer);
		chunkBuffer = malloc(bufferNeeded + 65536);
		chunkBufferCap = bufferNeeded + 65536;
	}
	unsigned char* buffer = chunkBuffer;
	float* floats = 0;
	if (!placing->isStored)
	{
		uint64_t floatsNeeded = columns * sizeof(float) + 1;
		if (floatsNeeded > chunkFloatsCap)
		{
			free(chunkFloats);
		chunkFloats = malloc(floatsNeeded + 65536);
			chunkFloatsCap = floatsNeeded + 65536;
		}
		floats = chunkFloats;
	}
	int isRead = buffer && (placing->isStored || floats) && (placing->read ?
		memcpy(buffer, placing->read + (first - placing->first) * rowBytes, count * rowBytes) != 0 :
		ReadFrom(placing->file, placing->start + first * rowBytes, buffer, count * rowBytes));
	if (!isRead)
		atomic_store(&placing->failed, 1);
	for (uint64_t row = first; isRead && row < first + count; row++)
	{
		unsigned char* target = placing->data + row * placing->stride;
		const unsigned char* source = buffer + (row - first) * rowBytes;
		if (placing->fours && placing->fours->isCompact)
		{
			Fourths(placing->fours, row, source, target, tensor->width);
			continue;
		}
		if (placing->fours)
			Unpacked(placing->fours, row, source, floats, tensor->width);
		else if (floats)
			Widened(source, tensor, floats, tensor->width);
		if (floats)
			Packed(target, floats, columns, 1, placing->bits);
		if (!floats && tensor->isNibbles)
			for (uint64_t column = 0; column < columns; column++)
				target[column] = (unsigned char)((int8_t)(source[column / 2] << (column % 2 ? 0 : 4)) >> 4);
		else if (!floats)
			memcpy(target, source, rowBytes);
		float* parts = (float*)(target + strict_packed_data(columns, 8));
		float scale = placing->scales ? placing->scales[row] : 0;
		if (placing->scales)
			for (uint64_t part = 0; part < (columns + 31) / 32; part++)
				parts[part] = scale;
	}
}

/* Reads the rows of tensor straight to data (stride bytes apart) with bits, as they are in the file when stored, in
   chunks on all threads, the failure or 0. */
static const char* Placed(double path, const Tensor* tensor, unsigned char* data, uint64_t stride, double bits,
	int isStored, const float* scales, const Fours* fours)
{
	uint64_t length = 0, fileSize = 0;
	if (IsRemote(strict_text(path)))
	{
		uint64_t start = DataStart(strict_text(path)), rowBytes = tensor->width * tensor->bytes;
		uint64_t slab = 67108864 / rowBytes > 1 ? 67108864 / rowBytes : 1;
		unsigned char* read = start ? malloc(slab * rowBytes + 1) : 0;
		Placing placing = { 0, tensor, 0, data, stride, bits, isStored, scales, fours };
		placing.read = read;
		atomic_store(&placing.failed, !read);
		for (uint64_t first = 0; !atomic_load(&placing.failed) && first < tensor->height; first += slab)
		{
			uint64_t count = tensor->height - first < slab ? tensor->height - first : slab;
			placing.first = first;
			placing.last = first + count;
			if (strict_webfile_read(strict_text(path), (double)(start + tensor->start + first * rowBytes),
				(double)(count * rowBytes), read))
				Parallel(PlacedChunk, &placing, (count + ChunkRows - 1) / ChunkRows);
			else
				atomic_store(&placing.failed, 1);
		}
		free(read);
		return atomic_load(&placing.failed) ? "TensorFileDataTruncated" : 0;
	}
	uint64_t start = DataStart(strict_text(path));
	if (!start)
	{
		FILE* file = Opened(strict_text(path), &length);
		if (!file)
			return "TensorFileNotReadable";
		fclose(file);
		start = 8 + length;
	}
	Placing placing = { OpenedFile(strict_text(path), &fileSize), tensor, start + tensor->start, data, stride, bits,
		isStored, scales, fours };
	if (!placing.file)
		return "TensorFileNotReadable";
	Parallel(PlacedChunk, &placing, (tensor->height + ChunkRows - 1) / ChunkRows);
	Closed(placing.file);
	return atomic_load(&placing.failed) ? "TensorFileDataTruncated" : 0;
}

/* The numbers of the tensor that entry describes: 32 bit floats for float tensors when bits is 32 (exact, the GPU keeps
   them as they are, read in chunks on all threads), doubles otherwise. */
STRICT_EXPORT double TensorFile_Floats(double self, double path, double entry, double bits)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor = { 0 };
	const char* failure = Described(strict_text(entry), &tensor);
	uint64_t total = tensor.height * tensor.width;
	Profiled("floats", tensor.height, tensor.width, ProfileNow());
	if (!failure && bits == 32 && tensor.bytes > 1)
	{
		unsigned char* data = 0;
		double result = strict_packed_numbers(total, 32, &data);
		failure = !data ? "TensorTooLarge" : Placed(path, &tensor, data, tensor.width * sizeof(float), 32, tensor.bytes == 4, 0, 0);
		return failure ? strict_fail(failure) : result;
	}
	float* floats = failure ? 0 : Read(path, &tensor, &failure);
	if (!floats)
		return strict_fail(failure);
	free(floatStorage);
	strict_numbers* numbers = malloc(sizeof(strict_numbers) + total * sizeof(double));
	floatStorage = (unsigned char*)numbers;
	numbers->count = (double)total;
	for (uint64_t index = 0; index < total; index++)
		numbers->items[index] = floats[index];
	free(floats);
	return strict_numbers_result(numbers);
}

/* The rows (or the columns when transposed) of the matrix that entry describes, each as numbers with bits 32 (exact for
   32 and 16 bit tensors), 17 (rounded to bfloat16, BF16 rows are read as they are), 16 (rounded to halves, F16 rows are
   read as they are) or 8 (quantized, strict_quantized). */
static double Matrix(double path, double entry, double bits, int transposed)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor;
	const char* failure = Described(strict_text(entry), &tensor);
	if (failure)
		return strict_fail(failure);
	if (tensor.rank != 2 || (bits != 8 && bits != 16 && bits != 17 && bits != 32))
		return strict_fail("OnlyMatricesOf8Or16Or17Or32BitsAreSupported");
	if (tensor.isNibbles)
		return strict_fail("I4TensorsAreOnlyReadAsQuantizedRows");
	uint64_t height = tensor.height, width = tensor.width, stride = 0;
	uint64_t outer = transposed ? width : height, inner = transposed ? height : width;
	int isStored = bits == 17 ? tensor.isBrain : !tensor.isBrain && bits > 8 && tensor.bytes * 8 == bits;
	float* floats = transposed ? Read(path, &tensor, &failure) : 0;
	if (failure)
		return strict_fail(failure);
	unsigned char* data = 0;
	double result = strict_rows(outer, inner, bits, &data, &stride);
	failure = !data ? "TensorTooLarge" : transposed ? 0 : Placed(path, &tensor, data, stride, bits, isStored, 0, 0);
	for (uint64_t row = 0; floats && !failure && row < outer; row++)
		Packed(data + row * stride, floats + row, inner, width, bits);
	free(floats);
	return failure ? strict_fail(failure) : result;
}

STRICT_EXPORT double TensorFile_Rows(double self, double path, double entry, double bits)
{
	double started = ProfileNow();
	double result = Matrix(path, entry, bits, 0);
	Profiled("rows", 0, 0, started);
	return result;
}

STRICT_EXPORT double TensorFile_Columns(double self, double path, double entry, double bits)
{
	return Matrix(path, entry, bits, 1);
}

/* numbers as rows of width 32 bit floats in one block like the rows of a file (strict_rows): rows a loop builds are an
   object each, H3's LoRA merged adaln rows (4.9 million of 8 numbers) made every collection visit them all */
STRICT_EXPORT double TensorFile_Shaped(double self, double numbers, double width)
{
	uint64_t total = strict_count(numbers), columns = width >= 1 ? (uint64_t)width : 0;
	if (columns == 0 || total % columns != 0)
		return strict_fail("ShapedWidthDoesNotFitTheNumbers");
	unsigned char* data = 0;
	uint64_t stride = 0;
	double result = strict_rows(total / columns, columns, 32, &data, &stride);
	if (!data)
		return strict_fail("TensorTooLarge");
	for (uint64_t row = 0; row < total / columns; row++)
		for (uint64_t column = 0; column < columns; column++)
			((float*)(data + row * stride))[column] = (float)strict_number(numbers, row * columns + column);
	return result;
}

/* The rows of an I8 matrix (tensor from entry) with one F32 scale per row (comfy_quant int8 rows, convrot keeps them
   rotated), read straight into 8 bit rows (bits, plus STRICT_KEPT_TOKENS) at data, stride bytes apart, whose per-32
   scales all are the scale of their row, so every number is exact: the result. */
static double QuantizedRows(double path, double entry, double scales, double bits, Tensor* tensor, unsigned char** data,
	uint64_t* stride, int isHosted, uint64_t first, uint64_t count)
{
	Tensor scaled;
	const char* failure = Described(strict_text(entry), tensor);
	if (!failure)
		failure = Described(strict_text(scales), &scaled);
	if (!failure && (!tensor->isSigned || tensor->rank != 2 || scaled.bytes != 4 || scaled.height * scaled.width != tensor->height))
		failure = "OnlyI8MatricesWithAnF32ScalePerRowAreSupported";
	float* factors = failure ? 0 : Read(path, &scaled, &failure);
	if (!factors)
		return strict_fail(failure);
#ifdef __EMSCRIPTEN__
	if (!isHosted && IsRemote(strict_text(path)))
	{
		uint64_t columns = tensor->width * (tensor->isNibbles ? 2 : 1);
		uint64_t codes = tensor->isNibbles ? tensor->width : strict_packed_data(columns, 8);
		int handle = strict_webfile_rows(strict_text(path), (double)(DataStart(strict_text(path)) + tensor->start),
			(double)tensor->height, (double)tensor->width, (double)codes, (double)codes, (double)columns, factors);
		free(factors);
		return handle ? strict_rows_web(tensor->height, columns, bits, handle, tensor->isNibbles) :
			strict_fail("TensorFileDataTruncated");
	}
#endif
	Tensor part = *tensor;
	uint64_t rows = first >= tensor->height ? 0 : tensor->height - first;
	part.height = count && count < rows ? count : rows;
	part.start = tensor->start + first * tensor->width * tensor->bytes;
	double profiled = ProfileNow();
	double result = strict_rows(part.height, tensor->width * (tensor->isNibbles ? 2 : 1), bits, data, stride);
	profileRows += ProfileNow() - profiled;
	profiled = ProfileNow();
	failure = !*data ? "TensorTooLarge" : Placed(path, &part, *data, *stride, bits, 1, factors + first, 0);
	profilePlaced += ProfileNow() - profiled;
	free(factors);
	return failure ? strict_fail(failure) : result;
}

STRICT_EXPORT double TensorFile_Quantized(double self, double path, double entry, double scales, double tokenBits)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor;
	unsigned char* data = 0;
	uint64_t stride = 0;
	double started = ProfileNow();
	double result = QuantizedRows(path, entry, scales, tokenBits > 8 ? 8 + STRICT_KEPT_TOKENS : 8, &tensor, &data, &stride, 0, 0,
		0);
	Profiled("quantized", tensor.height, tensor.width, started);
	return result;
}

/* count rows from first of TensorFile_Quantized kept on the host also when the file is read from a URL on the web
   (there the rows are otherwise only in a WebGPU buffer): for rows the program reads itself, like an embedding table,
   in parts, as words on the wasm stack that are no references kept a 1 GB block of it alive. */
STRICT_EXPORT double TensorFile_HostedPart(double self, double path, double entry, double scales, double first,
	double count)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor;
	unsigned char* data = 0;
	uint64_t stride = 0;
	return QuantizedRows(path, entry, scales, 8, &tensor, &data, &stride, 1, (uint64_t)first, (uint64_t)count);
}

/* The rows of an NVFP4 matrix (U8 entry of two numbers per byte, F8_E4M3 block scales, the F32 scale, the optional
   input factors of AWQ and the optional factors of its rows, the input factors of the next matrix: the up_proj rows
   take the factors of down_proj) as 8 bit rows (strict_quantized), above 8 token bits (rows only the GPU multiplies,
   plus STRICT_KEPT_TOKENS) without input factors as NVFP4 rows (format 5, Fourths): the bytes of the file, which stay
   on the GPU at 4.5 bits per number (the 15.7 GB H3 text encoder had 27 GB of 8 bit rows) and become 8 bit rows there
   just before a kernel reads them (strict_gpu_expanded); STRICT_FOURS_EIGHT=1 keeps 8 bit rows to compare them. */
STRICT_EXPORT double TensorFile_Fours(double self, double path, double entry, double scales, double scale, double inputs,
	double rowFactors, double tokenBits)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor, blocked, global, factors, rowed;
	const char* failure = Described(strict_text(entry), &tensor);
	if (!failure)
		failure = Described(strict_text(scales), &blocked);
	if (!failure)
		failure = Described(strict_text(scale), &global);
	int hasInputs = !failure && *strict_text(inputs);
	if (hasInputs)
		failure = Described(strict_text(inputs), &factors);
	int hasRowFactors = !failure && *strict_text(rowFactors);
	if (hasRowFactors)
		failure = Described(strict_text(rowFactors), &rowed);
	if (!failure && (tensor.bytes != 1 || tensor.isSigned || tensor.isEight || tensor.rank != 2 || !blocked.isEight ||
		blocked.height < (tensor.height + 127) / 128 * 128 || blocked.width < (tensor.width / 8 + 3) / 4 * 4 ||
		global.height * global.width != 1 || (hasInputs && factors.height * factors.width != 2 * tensor.width) ||
		(hasRowFactors && rowed.height * rowed.width != tensor.height)))
		failure = "OnlyNvfp4MatricesWithSwizzledBlockScalesAreSupported";
	double started = ProfileNow();
	unsigned char* blocks = failure ? 0 : Loaded(path, &blocked, 1, &failure);
	float* factor = blocks ? Read(path, &global, &failure) : 0;
	float* columns = factor && hasInputs ? Read(path, &factors, &failure) : 0;
	float* rows = factor && hasRowFactors ? Read(path, &rowed, &failure) : 0;
	profileRead += ProfileNow() - started;
	const char* eight = getenv("STRICT_FOURS_EIGHT");
	int isCompact = tokenBits > 8 && !hasInputs && !(eight && *eight == '1');
#ifdef __EMSCRIPTEN__
	isCompact = 0;
#endif
	unsigned char* data = 0;
	uint64_t stride = 0;
	double profiled = ProfileNow();
	double bits = (isCompact ? 5 : 8) + (tokenBits > 8 ? STRICT_KEPT_TOKENS : 0);
	double result = failure ? 0 : strict_rows(tensor.height, 2 * tensor.width, bits, &data, &stride);
	profileRows += ProfileNow() - profiled;
	Fours fours = { blocks, blocked.width, factor ? *factor : 0, columns, isCompact, rows };
	/* the block scales stay bytes, Unpacked widens them with the table: nothing else may have filled it yet (all NVFP4
	   weights of the text encoder were 0 when no 8 bit float was widened before) */
	EightTableInit();
	profiled = ProfileNow();
	if (!failure)
		failure = !data ? "TensorTooLarge" : Placed(path, &tensor, data, stride, 8, isCompact, 0, &fours);
	profilePlaced += ProfileNow() - profiled;
	Profiled(isCompact ? "fours compact" : "fours", tensor.height, tensor.width, started);
	free(rows);
	free(columns);
	free(factor);
	free(blocks);
	return failure ? strict_fail(failure) : result;
}

typedef struct { float* downs; float* ups; uint64_t width, rank, stride, height, seed; unsigned char* data; } Tuning;
enum { TileRows = 8, BlockColumns = 128 };

/* A number in [0, 1) for place in the stream of seed (splitmix64). */
static float Uniform(uint64_t seed, uint64_t place)
{
	uint64_t mixed = seed + (place + 1) * 0x9E3779B97F4A7C15ull;
	mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
	mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
	return (float)((mixed ^ (mixed >> 31)) >> 40) / 16777216.0f;
}

/* Quantizes row again from its tuned values with one scale, rounding stochastically like ComfyUI does for LoRAs on
   int8 weights, so small updates are kept on average. */
static void Requantized(const Tuning* tuning, uint64_t row, const float* values)
{
	uint64_t width = tuning->width;
	signed char* numbers = (signed char*)(tuning->data + row * tuning->stride);
	float* parts = (float*)((unsigned char*)numbers + strict_packed_data(width, 8)), largest = 0;
	for (uint64_t column = 0; column < width; column++)
		largest = fabsf(values[column]) > largest ? fabsf(values[column]) : largest;
	float scale = largest / 127;
	for (uint64_t column = 0; column < width; column++)
	{
		float scaled = scale > 0 ? values[column] / scale : 0, lower = floorf(scaled);
		float rounded = lower + (scaled - lower > Uniform(tuning->seed, row * width + column) ? 1 : 0);
		numbers[column] = (signed char)(rounded > 127 ? 127 : rounded < -127 ? -127 : rounded);
	}
	for (uint64_t part = 0; part < (width + 31) / 32; part++)
		parts[part] = scale;
}

/* Adds the update to the rows of one chunk that have one, a tile of rows and a block of columns at a time so the
   numbers being added to stay in the L1 cache and every part of a down row is read once per tile (rank 384 rows of
   5376 are 8 MB): 11.7 instead of 4.6 multiply-adds per ns on one Threadripper 3990X thread. */
static void TunedChunk(void* context, uint64_t chunk)
{
	Tuning* tuning = context;
	uint64_t width = tuning->width, rank = tuning->rank, first = chunk * ChunkRows, tuned[TileRows];
	uint64_t last = first + ChunkRows < tuning->height ? first + ChunkRows : tuning->height;
	float* values = malloc(TileRows * width * sizeof(float));
	for (uint64_t row = first; values && row < last;)
	{
		uint64_t count = 0;
		for (; row < last && count < TileRows; row++)
		{
			const float* up = tuning->ups + row * rank;
			int isTuned = 0;
			for (uint64_t part = 0; part < rank; part++)
				isTuned |= up[part] != 0;
			if (!isTuned)
				continue;
			const signed char* numbers = (const signed char*)(tuning->data + row * tuning->stride);
			float scale = ((const float*)((const unsigned char*)numbers + strict_packed_data(width, 8)))[0];
			float* target = values + count * width;
			for (uint64_t column = 0; column < width; column++)
				target[column] = numbers[column] * scale;
			tuned[count++] = row;
		}
		for (uint64_t start = 0; start < width; start += BlockColumns)
		{
			uint64_t end = start + BlockColumns < width ? start + BlockColumns : width;
			for (uint64_t part = 0; part < rank; part++)
			{
				const float* down = tuning->downs + part * width;
				for (uint64_t index = 0; index < count; index++)
				{
					float factor = tuning->ups[tuned[index] * rank + part], *target = values + index * width;
					for (uint64_t column = start; column < end; column++)
						target[column] += factor * down[column];
				}
			}
		}
		for (uint64_t index = 0; index < count; index++)
			Requantized(tuning, tuned[index], values + index * width);
	}
	free(values);
}

/* The rows of TensorFile_Quantized plus a low rank update: factors are the downs (rank x columns, in the rotated space
   of convrot rows), the ups (rows x rank) and one scale per rank; rows without an update stay as they are in the file. */
STRICT_EXPORT double TensorFile_Tuned(double self, double path, double entry, double scales, double factors)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor;
	unsigned char* data = 0;
	uint64_t stride = 0;
	double started = ProfileNow();
	double quantized = QuantizedRows(path, entry, scales, 8, &tensor, &data, &stride, 1, 0, 0);
	Profiled("tuned", tensor.height, tensor.width, started);
	if (strict_bits(quantized) >> 48 == 0x7FFC || strict_count(factors) == 0)
		return quantized;
	uint64_t height = tensor.height, width = tensor.width * (tensor.isNibbles ? 2 : 1);
	uint64_t rank = strict_count(factors) / (width + height + 1);
	if (rank == 0 || rank * (width + height + 1) != strict_count(factors))
		return strict_fail("TunedFactorsDoNotFitTheMatrix");
	Tuning tuning = { malloc(rank * width * sizeof(float)), malloc(rank * height * sizeof(float)), width, rank, stride,
		height, 1469598103934665603ull, data };
	if (!tuning.downs || !tuning.ups)
	{
		free(tuning.downs);
		free(tuning.ups);
		return strict_fail("TensorTooLarge");
	}
	uint64_t ranked = rank * (width + height);
	for (uint64_t index = 0; index < rank * width; index++)
		tuning.downs[index] = (float)strict_number(factors, index);
	for (uint64_t row = 0; row < height; row++)
		for (uint64_t part = 0; part < rank; part++)
			tuning.ups[row * rank + part] = (float)(strict_number(factors, rank * (width + row) + part) *
				strict_number(factors, ranked + part));
	for (const char* name = strict_text(entry); *name; name++)
		tuning.seed = (tuning.seed ^ (unsigned char)*name) * 1099511628211ull;
	Parallel(TunedChunk, &tuning, (height + ChunkRows - 1) / ChunkRows);
	free(tuning.downs);
	free(tuning.ups);
	return quantized;
}

/* Writes a new .safetensors file at path: the JSON header (padded to 8 bytes) and numbers as the 32 bit floats of the
   tensors it describes, their count. */
STRICT_EXPORT double TensorFile_Written(double self, double path, double header, double numbers)
{
	uint64_t count = strict_count(numbers), length = strlen(strict_text(header));
	static const char spaces[8] = "       ";
	unsigned char size[8];
	for (int index = 0; index < 8; index++)
		size[index] = (unsigned char)((length + (8 - length % 8) % 8) >> (8 * index));
	enum { chunk = 1 << 20 };
	float* floats = malloc(chunk * sizeof(float));
	FILE* file = floats ? fopen(strict_text(path), "wb") : 0;
	if (!file)
	{
		free(floats);
		return strict_fail("TensorFileNotWritable");
	}
	int complete = fwrite(size, 1, 8, file) == 8 && fwrite(strict_text(header), 1, length, file) == length &&
		fwrite(spaces, 1, (8 - length % 8) % 8, file) == (8 - length % 8) % 8;
	for (uint64_t start = 0; complete && start < count; start += chunk)
	{
		uint64_t part = count - start < chunk ? count - start : chunk;
		for (uint64_t index = 0; index < part; index++)
			floats[index] = (float)strict_number(numbers, start + index);
		complete = fwrite(floats, sizeof(float), part, file) == part;
	}
	complete = fclose(file) == 0 && complete;
	free(floats);
	return complete ? (double)count : strict_fail("TensorFileNotWritable");
}

/* Reads a whole file once on a thread of its own, 16 MB at a time, so its pages are in the file cache when a program
   reads them later: H3Video's text encoder process reads H3 and its LoRAs (24 GB) once it loaded itself, while it
   encodes on the GPU, and H3Video the VAEs while it samples. At normal priority and without the sequential scan hint:
   both put the pages at the end of the standby list, the first ones dropped (a 28 GB encoder, its 16 GB file and H3 do
   not fit into 64 GB together). The bytes it is going to read, 0 when it can not. */
enum { PrefetchBytes = 16 << 20 };
#ifdef _WIN32
static DWORD WINAPI Prefetching(void* context)
{
	HANDLE file = context;
	unsigned char* buffer = malloc(PrefetchBytes);
	DWORD read = 0;
	while (buffer && ReadFile(file, buffer, PrefetchBytes, &read, 0) && read > 0)
		read = 0;
	free(buffer);
	CloseHandle(file);
	return 0;
}
#elif !defined(__EMSCRIPTEN__)
static void* Prefetching(void* context)
{
	int file = (int)(intptr_t)context;
	unsigned char* buffer = malloc(PrefetchBytes);
	while (buffer && read(file, buffer, PrefetchBytes) > 0)
		;
	free(buffer);
	close(file);
	return 0;
}
#endif

STRICT_EXPORT double TensorFile_Prefetched(double self, double path)
{
#ifdef __EMSCRIPTEN__
	return 0;
#elif defined(_WIN32)
	HANDLE file = CreateFileA(strict_text(path), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
	if (file == INVALID_HANDLE_VALUE)
		return 0;
	LARGE_INTEGER size = { 0 };
	HANDLE thread = GetFileSizeEx(file, &size) ? CreateThread(0, 0, Prefetching, file, 0, 0) : 0;
	if (!thread)
	{
		CloseHandle(file);
		return 0;
	}
	CloseHandle(thread);
	return (double)size.QuadPart;
#else
	int file = open(strict_text(path), O_RDONLY);
	off_t size = file < 0 ? 0 : lseek(file, 0, SEEK_END);
	pthread_t thread;
	if (file < 0 || size <= 0 || lseek(file, 0, SEEK_SET) != 0 ||
		pthread_create(&thread, 0, Prefetching, (void*)(intptr_t)file) != 0)
	{
		if (file >= 0)
			close(file);
		return 0;
	}
	pthread_detach(thread);
	return (double)size;
#endif
}
