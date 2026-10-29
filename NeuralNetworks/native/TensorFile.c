#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "strict.h"
#include "Parallel.h"
#include "ThreadState.h"
static _Thread_local char* entry;
static _Thread_local unsigned char* floatStorage;
#ifdef _WIN32
#define strict_seek _fseeki64
#else
#include <sys/stat.h>
#define strict_seek fseeko
#endif
/* The headers of the files read last, each kept while its file has the same size and modification time: every tensor
   lookup read the whole header again (60-130 KB for a LoRA of Moss, about 5 lookups per merged linear). */
enum { HeaderCount = 8 };
typedef struct { char* path; uint64_t size, modified; char* text; } Header;
static _Thread_local Header headers[HeaderCount];
static _Thread_local int nextHeader;

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

/* The JSON header of the .safetensors file at path from headers or read into them, 0 when it can't be read. */
static const char* HeaderOf(const char* path, const char** failure)
{
	uint64_t size = 0, modified = 0, length = 0;
	*failure = "TensorFileNotReadable";
	if (!Stamped(path, &size, &modified))
		return 0;
	for (int index = 0; index < HeaderCount; index++)
		if (headers[index].path && !strcmp(headers[index].path, path) && headers[index].size == size &&
			headers[index].modified == modified)
			return headers[index].text;
	FILE* file = Opened(path, &length);
	if (!file || length > 100000000)
	{
		if (file)
			fclose(file);
		return 0;
	}
	char* text = malloc(length + 1);
	size_t read = text ? fread(text, 1, length, file) : 0;
	fclose(file);
	if (read != length)
	{
		free(text);
		*failure = "TensorFileHeaderTruncated";
		return 0;
	}
	text[length] = 0;
	Header* slot = &headers[nextHeader];
	nextHeader = (nextHeader + 1) % HeaderCount;
	free(slot->path);
	free(slot->text);
	size_t pathLength = strlen(path) + 1;
	slot->path = malloc(pathLength);
	memcpy(slot->path, path, pathLength);
	slot->size = size;
	slot->modified = modified;
	slot->text = text;
	return text;
}

STRICT_EXPORT double TensorFile_Header(double self, double path)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	const char* failure = 0;
	const char* text = HeaderOf(strict_text(path), &failure);
	return text ? strict_text_result(text) : strict_fail(failure);
}

/* The entry of the tensor name in the header of the file at path ("dtype":"F32","shape":[3],"data_offsets":[0,12]),
   empty when the header has none. */
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
	char* key = malloc(nameLength + 5);
	key[0] = '"';
	memcpy(key + 1, wanted, nameLength);
	memcpy(key + 1 + nameLength, "\":{", 4);
	const char* found = strstr(text, key);
	free(key);
	const char* open = found ? found + nameLength + 4 : "";
	const char* close = strchr(open, '}');
	size_t entryLength = close ? (size_t)(close - open) : strlen(open);
	free(entry);
	entry = malloc(entryLength + 1);
	memcpy(entry, open, entryLength);
	entry[entryLength] = 0;
	return strict_text_result(entry);
}

/* A tensor as its header entry describes it: "dtype":"F16","shape":[2,3],"data_offsets":[24,36]. */
typedef struct { uint64_t bytes, isBrain, isSigned, isEight, rank, height, width, start; } Tensor;

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
	tensor->start = range[0];
	if (range[1] < range[0] || range[1] - range[0] != tensor->height * tensor->width * tensor->bytes)
		return "TensorSizeMismatch";
	return 0;
}

/* An 8 bit float with 4 exponent bits (bias 7) and 3 mantissa bits, the block scales of NVFP4. */
static float Eight(unsigned char bits)
{
	int exponent = (bits >> 3) & 15, mantissa = bits & 7;
	float magnitude = exponent ? ldexpf(1 + mantissa / 8.0f, exponent - 7) : ldexpf(mantissa / 8.0f, -6);
	return bits & 128 ? -magnitude : magnitude;
}

/* Widens count numbers in the format of tensor at source to floats at target, which may start at or before source as
   long as the numbers end together (every number is read before a float overwrites it). */
static void Widened(const unsigned char* source, const Tensor* tensor, float* target, uint64_t count)
{
	if (tensor->bytes == 4)
		memmove(target, source, count * sizeof(float));
	for (uint64_t index = 0; tensor->bytes == 2 && tensor->isBrain && index < count; index++)
		target[index] = strict_brain_float((uint16_t)(source[2 * index] | source[2 * index + 1] << 8));
	for (uint64_t index = 0; tensor->bytes == 2 && !tensor->isBrain && index < count; index++)
		target[index] = strict_half_float((uint16_t)(source[2 * index] | source[2 * index + 1] << 8));
	for (uint64_t index = 0; tensor->bytes == 1 && index < count; index++)
		target[index] = tensor->isEight ? Eight(source[index]) :
			tensor->isSigned ? (float)(int8_t)source[index] : (float)source[index];
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
	unsigned char* numbers = (unsigned char*)floats + count * (sizeof(float) - tensor->bytes);
	strict_seek(file, (int64_t)(8 + length + tensor->start), SEEK_SET);
	size_t read = fread(numbers, (size_t)tensor->bytes, count, file);
	fclose(file);
	if (read != count)
	{
		free(floats);
		*failure = "TensorFileDataTruncated";
		return 0;
	}
	Widened(numbers, tensor, floats, count);
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
}

/* Packs count floats (stride apart) into the numbers at data with bits 32, 16 (rounded to halves) or 8
   (strict_quantized). */
static void Packed(unsigned char* data, const float* first, uint64_t count, uint64_t stride, double bits)
{
	if (bits == 8)
		strict_quantized(data, first, count, stride);
	for (uint64_t column = 0; bits != 8 && column < count; column++)
		if (bits == 16)
			((uint16_t*)data)[column] = strict_half(first[column * stride]);
		else
			((float*)data)[column] = first[column * stride];
}

/* NVFP4 rows: two 4 bit e2m1 numbers per byte (the first in the high nibble), each 16 scaled by its 8 bit float in the
   swizzled cuBLAS layout of blocks (tiles of 128 rows by 4 scales), all by scale and each column by its input factor. */
typedef struct { const float* blocks; uint64_t blockWidth; float scale; const float* inputs; } Fours;
static const float Nibbles[16] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6 };

static void Unpacked(const Fours* fours, uint64_t row, const unsigned char* source, float* target, uint64_t bytes)
{
	for (uint64_t column = 0; column < 2 * bytes; column++)
	{
		uint64_t block = column / 16;
		uint64_t tile = (row / 128) * (fours->blockWidth / 4) + block / 4;
		float scale = fours->blocks[tile * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + block % 4] * fours->scale;
		unsigned char nibble = column % 2 ? source[column / 2] & 15 : source[column / 2] >> 4;
		target[column] = Nibbles[nibble] * scale * (fours->inputs ? fours->inputs[column] : 1);
	}
}

enum { ChunkRows = 64 };
typedef struct { Opening file; const Tensor* tensor; uint64_t start; unsigned char* data; uint64_t stride; double bits;
	int isStored; const float* scales; const Fours* fours; atomic_int failed; } Placing;

/* Reads the rows of one chunk at once into a buffer and copies them to their place, converted unless stored, 8 bit rows
   with a scale each get it for all their 32 numbers. */
static void PlacedChunk(void* context, uint64_t chunk)
{
	Placing* placing = context;
	const Tensor* tensor = placing->tensor;
	uint64_t rowBytes = tensor->width * tensor->bytes, first = chunk * ChunkRows;
	uint64_t count = tensor->height - first < ChunkRows ? tensor->height - first : ChunkRows;
	uint64_t columns = placing->fours ? 2 * tensor->width : tensor->width;
	unsigned char* buffer = malloc(count * rowBytes + 1);
	float* floats = placing->isStored ? 0 : malloc(columns * sizeof(float) + 1);
	int isRead = buffer && (placing->isStored || floats) &&
		ReadFrom(placing->file, placing->start + first * rowBytes, buffer, count * rowBytes);
	if (!isRead)
		atomic_store(&placing->failed, 1);
	for (uint64_t row = first; isRead && row < first + count; row++)
	{
		unsigned char* target = placing->data + row * placing->stride;
		const unsigned char* source = buffer + (row - first) * rowBytes;
		if (placing->fours)
			Unpacked(placing->fours, row, source, floats, tensor->width);
		else if (floats)
			Widened(source, tensor, floats, tensor->width);
		if (floats)
			Packed(target, floats, columns, 1, placing->bits);
		else
			memcpy(target, source, rowBytes);
		float* parts = (float*)(target + strict_packed_data(tensor->width, 8));
		for (uint64_t part = 0; placing->scales && part < (tensor->width + 31) / 32; part++)
			parts[part] = placing->scales[row];
	}
	free(floats);
	free(buffer);
}

/* Reads the rows of tensor straight to data (stride bytes apart) with bits, as they are in the file when stored, in
   chunks on all threads, the failure or 0. */
static const char* Placed(double path, const Tensor* tensor, unsigned char* data, uint64_t stride, double bits,
	int isStored, const float* scales, const Fours* fours)
{
	uint64_t length = 0, fileSize = 0;
	FILE* file = Opened(strict_text(path), &length);
	if (!file)
		return "TensorFileNotReadable";
	fclose(file);
	Placing placing = { OpenedFile(strict_text(path), &fileSize), tensor, 8 + length + tensor->start, data, stride, bits,
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
   32 and 16 bit tensors), 16 (rounded to halves, F16 rows are read as they are) or 8 (quantized, strict_quantized). */
static double Matrix(double path, double entry, double bits, int transposed)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor;
	const char* failure = Described(strict_text(entry), &tensor);
	if (failure)
		return strict_fail(failure);
	if (tensor.rank != 2 || (bits != 8 && bits != 16 && bits != 32))
		return strict_fail("OnlyMatricesOf8Or16Or32BitsAreSupported");
	uint64_t height = tensor.height, width = tensor.width, stride = 0;
	uint64_t outer = transposed ? width : height, inner = transposed ? height : width;
	int isStored = !tensor.isBrain && bits > 8 && tensor.bytes * 8 == bits;
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
	return Matrix(path, entry, bits, 0);
}

STRICT_EXPORT double TensorFile_Columns(double self, double path, double entry, double bits)
{
	return Matrix(path, entry, bits, 1);
}

/* The rows of an I8 matrix (tensor from entry) with one F32 scale per row (comfy_quant int8 rows, convrot keeps them
   rotated), read straight into 8 bit rows (bits, plus STRICT_KEPT_TOKENS) at data, stride bytes apart, whose per-32
   scales all are the scale of their row, so every number is exact: the result. */
static double QuantizedRows(double path, double entry, double scales, double bits, Tensor* tensor, unsigned char** data,
	uint64_t* stride)
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
	double result = strict_rows(tensor->height, tensor->width, bits, data, stride);
	failure = !*data ? "TensorTooLarge" : Placed(path, tensor, *data, *stride, bits, 1, factors, 0);
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
	return QuantizedRows(path, entry, scales, tokenBits > 8 ? 8 + STRICT_KEPT_TOKENS : 8, &tensor, &data, &stride);
}

/* The rows of an NVFP4 matrix (U8 entry of two numbers per byte, F8_E4M3 block scales, the F32 scale and the optional
   input factors of AWQ) as 8 bit rows (strict_quantized, plus STRICT_KEPT_TOKENS above 8 token bits). */
STRICT_EXPORT double TensorFile_Fours(double self, double path, double entry, double scales, double scale, double inputs,
	double tokenBits)
{
	if (!ThreadReady())
		return strict_fail("NativeThreadStateFailed");
	Tensor tensor, blocked, global, factors;
	const char* failure = Described(strict_text(entry), &tensor);
	if (!failure)
		failure = Described(strict_text(scales), &blocked);
	if (!failure)
		failure = Described(strict_text(scale), &global);
	int hasInputs = !failure && *strict_text(inputs);
	if (hasInputs)
		failure = Described(strict_text(inputs), &factors);
	if (!failure && (tensor.bytes != 1 || tensor.isSigned || tensor.isEight || tensor.rank != 2 || !blocked.isEight ||
		blocked.height < (tensor.height + 127) / 128 * 128 || blocked.width < (tensor.width / 8 + 3) / 4 * 4 ||
		global.height * global.width != 1 || (hasInputs && factors.height * factors.width != 2 * tensor.width)))
		failure = "OnlyNvfp4MatricesWithSwizzledBlockScalesAreSupported";
	float* blocks = failure ? 0 : Read(path, &blocked, &failure);
	float* factor = blocks ? Read(path, &global, &failure) : 0;
	float* columns = factor && hasInputs ? Read(path, &factors, &failure) : 0;
	unsigned char* data = 0;
	uint64_t stride = 0;
	double result = failure ? 0 : strict_rows(tensor.height, 2 * tensor.width, tokenBits > 8 ? 8 + STRICT_KEPT_TOKENS : 8,
		&data, &stride);
	Fours fours = { blocks, blocked.width, factor ? *factor : 0, columns };
	if (!failure)
		failure = !data ? "TensorTooLarge" : Placed(path, &tensor, data, stride, 8, 0, 0, &fours);
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
	double quantized = QuantizedRows(path, entry, scales, 8, &tensor, &data, &stride);
	const strict_numbers* given = strict_numbers_of(factors);
	if (strict_bits(quantized) >> 48 == 0x7FFC || given->count == 0)
		return quantized;
	uint64_t height = tensor.height, width = tensor.width;
	uint64_t rank = (uint64_t)given->count / (width + height + 1);
	if (rank == 0 || rank * (width + height + 1) != (uint64_t)given->count)
		return strict_fail("TunedFactorsDoNotFitTheMatrix");
	Tuning tuning = { malloc(rank * width * sizeof(float)), malloc(rank * height * sizeof(float)), width, rank, stride,
		height, 1469598103934665603ull, data };
	if (!tuning.downs || !tuning.ups)
	{
		free(tuning.downs);
		free(tuning.ups);
		return strict_fail("TensorTooLarge");
	}
	const double* ranked = given->items + rank * (width + height);
	for (uint64_t index = 0; index < rank * width; index++)
		tuning.downs[index] = (float)given->items[index];
	for (uint64_t index = 0; index < rank * height; index++)
		tuning.ups[index] = (float)(given->items[rank * width + index] * ranked[index % rank]);
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
	const strict_numbers* given = strict_numbers_of(numbers);
	uint64_t count = (uint64_t)given->count, length = strlen(strict_text(header));
	static const char spaces[8] = "       ";
	unsigned char size[8];
	for (int index = 0; index < 8; index++)
		size[index] = (unsigned char)((length + (8 - length % 8) % 8) >> (8 * index));
	float* floats = malloc(count * sizeof(float) + 1);
	FILE* file = floats ? fopen(strict_text(path), "wb") : 0;
	if (!file)
	{
		free(floats);
		return strict_fail("TensorFileNotWritable");
	}
	for (uint64_t index = 0; index < count; index++)
		floats[index] = (float)given->items[index];
	int complete = fwrite(size, 1, 8, file) == 8 && fwrite(strict_text(header), 1, length, file) == length &&
		fwrite(spaces, 1, (8 - length % 8) % 8, file) == (8 - length % 8) % 8 &&
		fwrite(floats, sizeof(float), count, file) == count;
	complete = fclose(file) == 0 && complete;
	free(floats);
	return complete ? (double)count : strict_fail("TensorFileNotWritable");
}
