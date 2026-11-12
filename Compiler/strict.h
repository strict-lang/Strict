#ifndef STRICT_H
#define STRICT_H
#include <stdint.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#if defined(_WIN32) && defined(STRICT_LIBRARY)
#define STRICT_EXPORT __declspec(dllexport)
#elif defined(STRICT_LIBRARY)
#define STRICT_EXPORT __attribute__((visibility("default")))
#else
#define STRICT_EXPORT
#endif
/* Every argument and result is one double: a Number, or the bits of a Text or Numbers. */
typedef struct { double count; double items[]; } strict_numbers;
static inline uint64_t strict_bits(double value) { union { double number; uint64_t bits; } slot; slot.number = value; return slot.bits; }
static inline double strict_value(uint64_t bits) { union { double number; uint64_t bits; } slot; slot.bits = bits; return slot.number; }
/* Arguments are borrowed for the call only. */
static inline const void* strict_address(double value) { return (const void*)(uintptr_t)(strict_bits(value) & 0xFFFFFFFFFFFFull); }
static inline const char* strict_text(double value) { return (const char*)strict_address(value); }
/* A Numbers argument is the list Strict holds, read where it is: strict_count numbers, doubles at strict_items or, when
   that is 0, packed at strict_data in strict_format (32 floats, 16 halves, 17 bfloat16, 9 bytes, 33 4 byte lanes);
   strict_number reads each of them. Lists in other formats arrive as doubles. */
typedef struct { double length, holder; } strict_list;
typedef struct { double items, used, capacity, data; } strict_list_holder;
static inline uint64_t strict_count(double list)
{
	const strict_list* view = (const strict_list*)strict_address(list);
	return view ? (uint64_t)view->length : 0;
}
static inline const strict_list_holder* strict_holder(double list)
{
	const strict_list* view = (const strict_list*)strict_address(list);
	return view ? (const strict_list_holder*)strict_address(view->holder) : 0;
}
static inline int strict_format(double list)
{
	const strict_list_holder* holder = strict_holder(list);
	return holder && !strict_bits(holder->items) && holder->used >= 0 && holder->capacity < 0 ? (int)-holder->capacity : 0;
}
static inline const double* strict_items(double list)
{
	const strict_list_holder* holder = strict_holder(list);
	return holder && !strict_format(list) ? (const double*)strict_address(holder->items) : 0;
}
static inline const unsigned char* strict_data(double list)
{
	const strict_list_holder* holder = strict_holder(list);
	return holder ? (const unsigned char*)strict_address(holder->data) : 0;
}
/* The caller copies a returned Text or Numbers right away, it only has to live until the call returns. */
static inline double strict_text_result(const char* text) { return strict_value((uint64_t)(uintptr_t)text); }
static inline double strict_numbers_result(const strict_numbers* numbers) { return strict_value((uint64_t)(uintptr_t)numbers); }
/* Numbers stored with fewer bits: count values as 32 bit floats, 16 bit halves or 8 bit quantized (bits is 32, 16 or 8),
   returned with strict_packed_result. A List(Numbers) result is a strict_numbers whose items are Numbers or packed results,
   one per row. 8 bits are a signed byte per number followed (at a multiple of 4 bytes) by a float scale for every 32
   numbers, a number is its byte times the scale of its 32 (Q8_0 of ggml with the scales apart, strict_quantized). 4 bits
   are the Q4_K blocks of ggml (144 bytes for 256 numbers), 6 bits the Q6_K blocks without their half scale d (208
   bytes for 256 numbers) followed by the d of every block, so each block starts 16 byte aligned, both decode like ggml.
   3 bits are ternary weights, a number is (code - 1) times the half scale of its 128, in the PTQ1_0 blocks of Prism
   (1.75 bits per number, 28 bytes for 128): 26 bytes of codes, 5 per byte as a fraction of 243 (code n of a byte x is
   ((x * 3^n) % 256 * 3) / 256), bytes 0 to 15 give numbers 0 to 79 (code n of byte b is number 16n + b), bytes 16 to 23
   numbers 80 to 119 (80 + 8n + b), bytes 24 and 25 four codes each for 120 to 127 (120 + 2n + b), then the half scale. */
typedef struct { double count; double bits; unsigned char data[]; } strict_packed;
static inline double strict_packed_result(const strict_packed* packed) { return strict_value(0x7FF9000000000000ull | (uint64_t)(uintptr_t)packed); }
/* Added to the bits of a packed row: dot products with it on the GPU keep their tokens unquantized. */
#define STRICT_KEPT_TOKENS 256
/* Format 5 are NVFP4 rows: two 4 bit e2m1 numbers per byte (the first in the high nibble) in groups of 32, then an 8
   bit float (E4M3) scale per 16 numbers, then the 32 bit factor of the row (4 byte aligned) */
static inline uint64_t strict_packed_data(uint64_t count, uint64_t bits)
{
	return bits == 4 ? (count + 255) / 256 * 144 : bits == 6 ? (count + 255) / 256 * 208 : bits == 3 ? (count + 127) / 128 * 28 :
		bits == 5 ? (count + 31) / 32 * 16 : (count + 3) / 4 * 4;
}
static inline uint64_t strict_packed_bytes(uint64_t count, uint64_t bits)
{
	uint64_t data = strict_packed_data(count, bits);
	return bits == 8 ? data + (count + 31) / 32 * 4 : bits == 6 ? data + ((count + 255) / 256 * 2 + 3) / 4 * 4 :
		bits == 5 ? ((count + 31) / 32 * 18 + 3) / 4 * 4 + 4 : bits == 4 || bits == 3 ? data : count * bits / 8;
}
/* Packs count floats (stride apart) into the 8 bit numbers at data, the scale of 32 numbers is their largest magnitude
   / 127, numbers round half away from zero (adding the float below 0.5 before truncating). */
static inline void strict_quantized(unsigned char* data, const float* values, uint64_t count, uint64_t stride)
{
	int8_t* bytes = (int8_t*)data;
	float* scales = (float*)(data + (count + 3) / 4 * 4);
	uint64_t first = 0;
#ifdef __AVX2__
	const __m256 sign = _mm256_set1_ps(-0.0f), half = _mm256_set1_ps(0.49999997f);
	for (; stride == 1 && first + 32 <= count; first += 32)
	{
		__m256 parts[4], most = _mm256_setzero_ps();
		for (int part = 0; part < 4; part++)
		{
			parts[part] = _mm256_loadu_ps(values + first + part * 8);
			most = _mm256_max_ps(most, _mm256_andnot_ps(sign, parts[part]));
		}
		__m128 four = _mm_max_ps(_mm256_castps256_ps128(most), _mm256_extractf128_ps(most, 1));
		four = _mm_max_ps(four, _mm_movehl_ps(four, four));
		float largest = _mm_cvtss_f32(_mm_max_ss(four, _mm_movehdup_ps(four)));
		scales[first / 32] = largest / 127;
		__m256 inverse = _mm256_set1_ps(largest == 0 ? 0 : 127 / largest);
		__m256i rounded[4];
		for (int part = 0; part < 4; part++)
			rounded[part] = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(parts[part], inverse),
				_mm256_or_ps(half, _mm256_and_ps(sign, parts[part]))));
		__m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(rounded[0], rounded[1]),
			_mm256_packs_epi32(rounded[2], rounded[3]));
		_mm256_storeu_si256((__m256i*)(bytes + first),
			_mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
	}
#endif
	for (; first < count; first += 32)
	{
		uint64_t end = first + 32 < count ? first + 32 : count;
		float largest = 0;
		for (uint64_t index = first; index < end; index++)
		{
			float magnitude = values[index * stride] < 0 ? -values[index * stride] : values[index * stride];
			largest = magnitude > largest ? magnitude : largest;
		}
		scales[first / 32] = largest / 127;
		float inverse = largest == 0 ? 0 : 127 / largest;
		for (uint64_t index = first; index < end; index++)
		{
			float scaled = values[index * stride] * inverse;
			bytes[index] = (int8_t)(scaled + (scaled < 0 ? -0.49999997f : 0.49999997f));
		}
	}
}
/* The nearest 16 bit float (IEEE half, ties to even) of a 32 bit float. */
static inline uint16_t strict_half(float number)
{
	union { float number; uint32_t bits; } slot = { number };
	uint32_t sign = slot.bits & 0x80000000u, magnitude = slot.bits ^ sign, half;
	if (magnitude >= 0x47800000u)
		half = magnitude > 0x7F800000u ? 0x7E00 : 0x7C00;
	else if (magnitude < 0x38800000u)
	{
		union { uint32_t bits; float number; } small = { magnitude };
		small.number += 0.5f;
		half = small.bits - 0x3F000000u;
	}
	else
		half = (magnitude + 0xC8000FFFu + ((magnitude >> 13) & 1)) >> 13;
	return (uint16_t)(half | sign >> 16);
}
/* strict_half of count numbers stride apart, 8 at once with F16C (its NaNs replaced by the quiet NaN of strict_half). */
static inline void strict_halves(uint16_t* halves, const float* numbers, uint64_t count, uint64_t stride)
{
	uint64_t index = 0;
#if defined(__F16C__) && defined(__AVX2__)
	for (; stride == 1 && index + 8 <= count; index += 8)
	{
		__m256 value = _mm256_loadu_ps(numbers + index), isNan = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
		__m128i half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT);
		if (_mm256_movemask_ps(isNan))
		{
			__m256i nans = _mm256_or_si256(_mm256_srli_epi32(_mm256_castps_si256(_mm256_and_ps(value,
				_mm256_set1_ps(-0.0f))), 16), _mm256_set1_epi32(0x7E00)), mask = _mm256_castps_si256(isNan);
			half = _mm_blendv_epi8(half, _mm_packus_epi32(_mm256_castsi256_si128(nans), _mm256_extracti128_si256(nans, 1)),
				_mm_packs_epi32(_mm256_castsi256_si128(mask), _mm256_extracti128_si256(mask, 1)));
		}
		_mm_storeu_si128((__m128i*)(halves + index), half);
	}
#endif
	for (; index < count; index++)
		halves[index] = strict_half(numbers[index * stride]);
}
/* The nearest bfloat16 (ties to even, a NaN stays a quiet NaN) of a 32 bit float. */
static inline uint16_t strict_brain(float number)
{
	union { float number; uint32_t bits; } slot = { number };
	if ((slot.bits & 0x7FFFFFFFu) > 0x7F800000u)
		return (uint16_t)(slot.bits >> 16 | 0x40);
	return (uint16_t)((slot.bits + 0x7FFFu + (slot.bits >> 16 & 1)) >> 16);
}
/* The 32 bit float of a 16 bit float (IEEE half) and of a bfloat16 (the upper half of a 32 bit float). */
static inline float strict_half_float(uint16_t half)
{
	uint32_t sign = (half & 0x8000u) << 16, exponent = (half >> 10) & 31, mantissa = half & 1023;
	union { uint32_t bits; float value; } single;
	if (exponent == 0)
		return (sign ? -1.0f : 1.0f) * (float)mantissa / 16777216.0f;
	single.bits = sign | (exponent == 31 ? 0x7F800000u : (exponent + 112) << 23) | mantissa << 13;
	return single.value;
}
static inline float strict_brain_float(uint16_t brain)
{
	union { uint32_t bits; float value; } single = { (uint32_t)brain << 16 };
	return single.value;
}
/* The number at index of a Numbers argument in any of its formats. */
static inline double strict_number(double list, uint64_t index)
{
	const double* items = strict_items(list);
	const unsigned char* data = strict_data(list);
	if (items)
		return items[index];
	switch (strict_format(list))
	{
	case 32: return ((const float*)data)[index];
	case 16: return strict_half_float(((const uint16_t*)data)[index]);
	case 17: return strict_brain_float(((const uint16_t*)data)[index]);
	case 33: return ((const uint32_t*)data)[index];
	default: return data[index];
	}
}
/* Returns an Error with this name, the name must live as long as the program. */
static inline double strict_fail(const char* name) { return strict_value(0x7FFC000000000000ull | (uint64_t)(uintptr_t)name); }
/* Creates the missing folders of a path a native is about to write, like File writes do. */
#ifdef _WIN32
#include <direct.h>
#define strict_mkdir(folder) _mkdir(folder)
#else
#include <sys/stat.h>
#define strict_mkdir(folder) mkdir(folder, 0777)
#endif
static inline void strict_folders(const char* path)
{
	char folder[1024];
	for (int index = 1; path[index] && index < (int)sizeof folder; index++)
		if (path[index] == '/' || path[index] == '\\')
		{
			for (int copied = 0; copied < index; copied++)
				folder[copied] = path[copied];
			folder[index] = 0;
			strict_mkdir(folder);
		}
}
/* Room for count rows of elements numbers with bits (plus STRICT_KEPT_TOKENS) as a List(Numbers) result: the caller
   writes the packed numbers of each row at data + row * stride, then returns the result (data is 0 when there is not
   enough memory). A native build places them where the runtime keeps its rows, a library for the VM in one buffer it
   keeps until the next call. */
#ifdef STRICT_LIBRARY
#include <stdlib.h>
static inline double strict_rows(uint64_t count, uint64_t elements, double bits, unsigned char** data, uint64_t* stride)
{
	static _Thread_local double* rows;
	static _Thread_local unsigned char* storage;
	static _Thread_local uint64_t capacity;
	uint64_t size = (sizeof(strict_packed) + strict_packed_bytes(elements, (uint64_t)bits & 255) + 15) / 16 * 16;
	free(rows);
	rows = malloc((1 + count) * sizeof(double));
	if (count * size + 1 > capacity)
	{
		free(storage);
		storage = malloc(count * size + 1);
		capacity = storage ? count * size + 1 : 0;
	}
	*data = 0;
	if (!rows || !storage)
		return strict_fail("NotEnoughMemory");
	rows[0] = (double)count;
	for (uint64_t row = 0; row < count; row++)
	{
		strict_packed* packed = (strict_packed*)(storage + row * size);
		packed->count = (double)elements;
		packed->bits = bits;
		rows[1 + row] = strict_packed_result(packed);
	}
	*data = storage + sizeof(strict_packed);
	*stride = size;
	return strict_numbers_result((strict_numbers*)rows);
}
/* Room for count numbers with bits as a Numbers result like strict_rows, the caller writes them to data. */
static inline double strict_packed_numbers(uint64_t count, double bits, unsigned char** data)
{
	static _Thread_local strict_packed* packed;
	free(packed);
	packed = malloc(sizeof(strict_packed) + strict_packed_bytes(count, (uint64_t)bits & 255) + 1);
	*data = packed ? packed->data : 0;
	if (!packed)
		return strict_fail("NotEnoughMemory");
	packed->count = (double)count;
	packed->bits = bits;
	return strict_packed_result(packed);
}
#else
double strict_rows(uint64_t count, uint64_t elements, double bits, unsigned char** data, uint64_t* stride);
double strict_packed_numbers(uint64_t count, double bits, unsigned char** data);
#endif
#endif
