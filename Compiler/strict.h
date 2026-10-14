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
/* Numbers stored with fewer bits: count values as 32 bit floats or 16 bit halves (bits is 32 or 16), returned with
   strict_packed_result. A List(Numbers) result is a strict_numbers whose items are Numbers or packed results, one per row. */
typedef struct { double count; double bits; unsigned char data[]; } strict_packed;
static inline double strict_packed_result(const strict_packed* packed) { return strict_value(0x7FF9000000000000ull | (uint64_t)(uintptr_t)packed); }
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
/* Returns an Error with this name, the name must live as long as the program. */
static inline double strict_fail(const char* name) { return strict_value(0x7FFC000000000000ull | (uint64_t)(uintptr_t)name); }
#endif
