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