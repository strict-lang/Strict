namespace Strict.Language;

public static class NumberExtensions
{
	public static bool IsNameLengthWithinLimit(this int length) =>
		length is >= Limit.NameMinLimit and <= Limit.NameMaxLimit;
}