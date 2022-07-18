using System;
using System.Linq;

namespace Strict.Language;

/// <summary>
/// https://strict.dev/docs/Keywords
/// </summary>
public static class BinaryOperator
{
	public const string Plus = "+";
	public const string Minus = "-";
	public const string Multiply = "*";
	public const string Divide = "/";
	public const string Modulate = "%";
	public const string Smaller = "<";
	public const string Greater = ">";
	public const string SmallerOrEqual = "<=";
	public const string GreaterOrEqual = ">=";
	public const string Is = "is";
	public const string As = "as";
	public const string To = "to";
	public const string And = "and";
	public const string Or = "or";
	public const string Xor = "xor";
	public static bool IsOperator(this string name) => All.Contains(name);

	public static string? FindFirstOperator(this string line) =>
		All.FirstOrDefault(l => line.Contains(" " + l + " ")); // TODO: Performance test with 1 Million lines and Optimize line.Contains with for loop

	private const string AnySingleCharacterOperator = Plus + Minus + Multiply + Divide + Modulate + Smaller + Greater;
	private static readonly string[] MultiCharacterOperatorsWithSpaces =
	{
		' ' + SmallerOrEqual + ' ', ' ' + GreaterOrEqual + ' ', ' ' + Is + ' ', ' ' + And + ' ',
		' ' + Or + ' ', ' ' + Xor + ' ', ' ' + As + ' ', ' ' + To + ' '
	};

	//TODO: do a performance test here for this!
	public static bool IsOperator(this ReadOnlyMemory<char> line)
	{
		var span = line.Span;
		if (span.IndexOfAny(AnySingleCharacterOperator) >= 0)
			return true;
		foreach (var multiCharacterOperator in MultiCharacterOperatorsWithSpaces)
			if (span.IndexOf(multiCharacterOperator) >= 0)
				return true;
		return false;
	}

	private static readonly string[] All =
	{
		Plus, Minus, Multiply, Divide, Modulate, Smaller, Greater, SmallerOrEqual, GreaterOrEqual,
		Is, And, Or, Xor, As, To
	};
	private static readonly string[] Arithmetic = { Plus, Minus, Multiply, Divide, Modulate };
	private static readonly string[] Comparison =
	{
		Is, Smaller, Greater, SmallerOrEqual, GreaterOrEqual
	};
	private static readonly string[] Logical = { And, Or, Xor };
	private static readonly string[] Conversions = { As, To };
}