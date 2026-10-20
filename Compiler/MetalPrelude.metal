// Kernels translated from the LLVM IR of NativeRuntimeDevice.mlir and the loops of a program by MetalShader.strict
#include <metal_stdlib>
using namespace metal;
// Metal has no doubles: their bits stay in a ulong, loads and stores move them, these convert them (nearest even)
float strict_float(ulong bits)
{
	uint sign = (uint)(bits >> 63) << 31;
	int exponent = (int)((bits >> 52) & 0x7FF);
	ulong mantissa = bits & 0xFFFFFFFFFFFFFul;
	if (exponent == 0x7FF)
		return as_type<float>(sign | 0x7F800000u | (mantissa != 0 ? 0x400000u : 0u));
	int single = exponent - 896;
	if (single >= 0xFF)
		return as_type<float>(sign | 0x7F800000u);
	int shift = single > 0 ? 29 : 30 - single;
	if (shift > 53)
		return as_type<float>(sign);
	ulong full = exponent == 0 ? mantissa : mantissa | (1ul << 52);
	ulong kept = full >> shift;
	ulong rest = full & ((1ul << shift) - 1);
	ulong middle = 1ul << (shift - 1);
	uint rounded = (uint)kept + ((rest > middle || (rest == middle && (kept & 1))) ? 1u : 0u);
	return as_type<float>(sign | (single > 0 ? ((uint)single << 23) + (rounded - 0x800000u) : rounded));
}
ulong strict_double(float value)
{
	uint bits = as_type<uint>(value);
	ulong sign = (ulong)(bits >> 31) << 63;
	int exponent = (int)((bits >> 23) & 0xFF);
	ulong mantissa = bits & 0x7FFFFFu;
	if (exponent == 0xFF)
		return sign | 0x7FF0000000000000ul | (mantissa << 29);
	if (exponent == 0 && mantissa == 0)
		return sign;
	if (exponent == 0)
		for (exponent = 1; (mantissa & 0x800000u) == 0; exponent--)
			mantissa <<= 1;
	return sign | ((ulong)(exponent + 896) << 52) | ((mantissa & 0x7FFFFFu) << 29);
}
int strict_saturated(float value)
{
	if (isnan(value))
		return 0;
	return value >= 2147483647.0f ? 2147483647 : value <= -2147483648.0f ? -2147483647 - 1 : (int)value;
}
int strict_rounded(float value) { return (int)rint(value); }
template<typename T> T strict_shuffled(int mask, T value, int offset, int clamp) { return simd_shuffle_xor(value, (ushort)offset); }
