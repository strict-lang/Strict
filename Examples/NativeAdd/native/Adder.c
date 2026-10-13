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
