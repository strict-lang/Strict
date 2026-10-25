#ifndef STRICT_H
#define STRICT_H
#include <stdint.h>
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
static inline const char* strict_text(double value) { return (const char*)(uintptr_t)(strict_bits(value) & 0xFFFFFFFFFFFFull); }
static inline const strict_numbers* strict_numbers_of(double value) { return (const strict_numbers*)(uintptr_t)(strict_bits(value) & 0xFFFFFFFFFFFFull); }
/* The caller copies a returned Text or Numbers right away, it only has to live until the call returns. */
static inline double strict_text_result(const char* text) { return strict_value((uint64_t)(uintptr_t)text); }
static inline double strict_numbers_result(const strict_numbers* numbers) { return strict_value((uint64_t)(uintptr_t)numbers); }
/* Numbers stored with fewer bits: count values as 32 bit floats, 16 bit halves or 8 bit quantized (bits is 32, 16 or 8),
   returned with strict_packed_result. A List(Numbers) result is a strict_numbers whose items are Numbers or packed results,
   one per row. 8 bits are a signed byte per number followed (at a multiple of 4 bytes) by a float scale for every 32
   numbers, a number is its byte times the scale of its 32 (Q8_0 of ggml with the scales apart, strict_quantized). 4 bits
   are the Q4_K blocks of ggml (144 bytes for 256 numbers), 6 bits the Q6_K blocks without their half scale d (208
   bytes for 256 numbers) followed by the d of every block, so each block starts 16 byte aligned, both decode like ggml.
   2 bits are ternary weights: 32 bytes of codes for every 128 numbers (in each 64, bits 2k of byte b are number 16k + b)
   followed by a half scale for every 128, a number is (code - 1) times its scale. */
typedef struct { double count; double bits; unsigned char data[]; } strict_packed;
static inline double strict_packed_result(const strict_packed* packed) { return strict_value(0x7FF9000000000000ull | (uint64_t)(uintptr_t)packed); }
static inline uint64_t strict_packed_data(uint64_t count, uint64_t bits)
{
	return bits == 4 ? (count + 255) / 256 * 144 : bits == 6 ? (count + 255) / 256 * 208 : bits == 2 ? (count + 127) / 128 * 32 :
		(count + 3) / 4 * 4;
}
static inline uint64_t strict_packed_bytes(uint64_t count, uint64_t bits)
{
	uint64_t data = strict_packed_data(count, bits);
	return bits == 8 ? data + (count + 31) / 32 * 4 : bits == 6 ? data + ((count + 255) / 256 * 2 + 3) / 4 * 4 : bits == 4 ? data :
		bits == 2 ? data + ((count + 127) / 128 * 2 + 3) / 4 * 4 : count * bits / 8;
}
/* Packs count floats (stride apart) into 8 bits, the scale of 32 numbers is their largest magnitude / 127. */
static inline void strict_quantized(strict_packed* packed, const float* values, uint64_t count, uint64_t stride)
{
	int8_t* bytes = (int8_t*)packed->data;
	float* scales = (float*)(packed->data + (count + 3) / 4 * 4);
	packed->count = (double)count;
	packed->bits = 8;
	for (uint64_t first = 0; first < count; first += 32)
	{
		uint64_t end = first + 32 < count ? first + 32 : count;
		float largest = 0;
		for (uint64_t index = first; index < end; index++)
		{
			float magnitude = values[index * stride] < 0 ? -values[index * stride] : values[index * stride];
			largest = magnitude > largest ? magnitude : largest;
		}
		float scale = largest / 127;
		scales[first / 32] = scale;
		for (uint64_t index = first; index < end; index++)
		{
			float scaled = scale == 0 ? 0 : values[index * stride] / scale;
			bytes[index] = (int8_t)(scaled < 0 ? scaled - 0.5f : scaled + 0.5f);
		}
	}
}
/* The nearest 16 bit float (IEEE half, ties to even) of a 32 bit float. */
static inline uint16_t strict_half(float number)
{
	union { float number; uint32_t bits; } slot = { number };
	uint32_t sign = (slot.bits >> 16) & 0x8000, exponent = (slot.bits >> 23) & 0xFF, mantissa = slot.bits & 0x7FFFFF;
	int biased = (int)exponent - 112;
	if (exponent == 0xFF)
		return (uint16_t)(sign | 0x7C00 | (mantissa ? 0x200 : 0));
	if (biased >= 31)
		return (uint16_t)(sign | 0x7C00);
	if (biased < -10)
		return (uint16_t)sign;
	uint32_t shift = biased > 0 ? 13 : (uint32_t)(14 - biased);
	uint32_t significand = biased > 0 ? mantissa : mantissa | 0x800000;
	uint32_t half = (biased > 0 ? (uint32_t)biased << 10 : 0) | (significand >> shift);
	uint32_t rest = significand & ((1u << shift) - 1), middle = 1u << (shift - 1);
	return (uint16_t)(sign | (half + (rest > middle || (rest == middle && (half & 1)))));
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
/* Returns an Error with this name, the name must live as long as the program. */
static inline double strict_fail(const char* name) { return strict_value(0x7FFC000000000000ull | (uint64_t)(uintptr_t)name); }
#endif
