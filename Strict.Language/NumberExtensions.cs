namespace Strict.Language;

public static class NumberExtensions
{
	public static bool IsWithinLimit(this int length) =>
		length is >= Limit.NameMinLimit and <= Limit.NameMaxLimit;
}