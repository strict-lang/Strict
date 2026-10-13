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

STRICT_EXPORT double Adder_Failing(double self)
{
	return strict_fail("NativeFailure");
}

/* ponytail: 16 counters per process; a handle table would grow when a real package needs more. */
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
