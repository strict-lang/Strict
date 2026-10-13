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
/* Returns an Error with this name, the name must live as long as the program. */
static inline double strict_fail(const char* name) { return strict_value(0x7FFC000000000000ull | (uint64_t)(uintptr_t)name); }
#endif
