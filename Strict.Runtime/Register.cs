namespace Strict.Runtime;

/// <summary>
/// 16 general purpose registers (R0-R15) for use in instructions, normally only very few are
/// needed (2-3), but if things are nested we might need keep track of a comparison result in an
/// if statement for the else part later on. Return value is always the last used register.
/// </summary>
public enum Register
{
	R0,
	R1,
	R2,
	R3,
	R4,
	R5,
	R6,
	R7,
	R8,
	R9,
	R10,
	R11,
	R12,
	R13,
	R14,
	R15
}