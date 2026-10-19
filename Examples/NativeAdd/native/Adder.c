#include <stdio.h>
#include "strict.h"

STRICT_EXPORT double Adder_Add(double self, double left, double right)
{
	return left + right;
}

STRICT_EXPORT double Adder_Greeting(double self, double name)
{
	static char greeting[256];
	snprintf(greeting, sizeof greeting, "Hello %s", strict_text(name));
	return strict_text_result(greeting);
}

STRICT_EXPORT double Adder_Total(double self, double numbers)
{
	const strict_numbers* list = strict_numbers_of(numbers);
	double total = 0;
	for (int index = 0; index < (int)list->count; index++)
		total += list->items[index];
	return total;
}

STRICT_EXPORT double Adder_Doubled(double self, double numbers)
{
	static double storage[1 + 64];
	strict_numbers* doubled = (strict_numbers*)storage;
	const strict_numbers* list = strict_numbers_of(numbers);
	doubled->count = list->count < 64 ? list->count : 64;
	for (int index = 0; index < (int)doubled->count; index++)
		doubled->items[index] = list->items[index] * 2;
	return strict_numbers_result(doubled);
}

STRICT_EXPORT double Adder_Halved(double self, double numbers)
{
	static unsigned char storage[sizeof(strict_packed) + 64 * 2];
	strict_packed* halves = (strict_packed*)storage;
	const strict_numbers* list = strict_numbers_of(numbers);
	uint16_t* data = (uint16_t*)halves->data;
	halves->count = list->count < 64 ? list->count : 64;
	halves->bits = 16;
	for (int index = 0; index < (int)halves->count; index++)
		data[index] = strict_half((float)list->items[index]);
	return strict_packed_result(halves);
}

/* Rows of columns numbers as 32 bit floats, at most 8 rows of 8. */
STRICT_EXPORT double Adder_Split(double self, double numbers, double columns)
{
	static double rows[1 + 8];
	static unsigned char storage[8][sizeof(strict_packed) + 8 * 4];
	const strict_numbers* list = strict_numbers_of(numbers);
	int width = columns < 1 ? 1 : columns > 8 ? 8 : (int)columns;
	rows[0] = (int)list->count / width < 8 ? (int)list->count / width : 8;
	for (int row = 0; row < (int)rows[0]; row++)
	{
		strict_packed* packed = (strict_packed*)storage[row];
		packed->count = width;
		packed->bits = 32;
		for (int column = 0; column < width; column++)
			((float*)packed->data)[column] = (float)list->items[row * width + column];
		rows[1 + row] = strict_packed_result(packed);
	}
	return strict_numbers_result((strict_numbers*)rows);
}

STRICT_EXPORT double Adder_Failing(double self)
{
	return strict_fail("NativeFailure");
}

/* A NaN whose payload bits look like an Error tag, data read from files can contain such values. */
STRICT_EXPORT double Adder_Undefined(double self)
{
	union { unsigned long long bits; double number; } value = { 0x7FFA000000000001ull };
	return value.number;
}

/* 16 counters per process; a handle table would grow when a real package needs more. */
static double counters[16][2];
static int used;

STRICT_EXPORT double Counter_from(double self, double start, double step)
{
	if (used == 16)
		return strict_fail("TooManyCounters");
	counters[used][0] = start;
	counters[used][1] = step;
	return ++used;
}

STRICT_EXPORT double Counter_Next(double self)
{
	int index = (int)self - 1;
	if (index < 0 || index >= used)
		return strict_fail("InvalidCounter");
	return counters[index][0] += counters[index][1];
}
