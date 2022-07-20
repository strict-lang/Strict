using System;
using System.Linq;
using System.Runtime.CompilerServices;

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
	public const string Power = "^";
	public const string Modulate = "%";
	public const string Smaller = "<";
	public const string Greater = ">";
	public const string SmallerOrEqual = "<=";
	public const string GreaterOrEqual = ">=";
	public const string Is = "is";
	public const string To = "to";
	//TODO: discuss in
	public const string And = "and";
	public const string Or = "or";
	public const string Xor = "xor";
	//TODO: remove
	public static bool IsOperator(this string name) => All.Contains(name);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsSingleCharacterOperator(this char tokenFirstCharacter) =>
		AnySingleCharacterOperator.Contains(tokenFirstCharacter);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsMultiCharacterOperator(this string name)
	{
		foreach (var checkOperator in MultiCharacterOperators)
			if (checkOperator == name)
				return true;
		return false;
	}

	private static readonly string[] MultiCharacterOperators =
	{
		SmallerOrEqual, GreaterOrEqual, Is, And, Or, Xor, To
	};

	public static string? FindFirstOperator(this string line) =>
		All.FirstOrDefault(l => line.Contains(" " + l + " "));

	private const string AnySingleCharacterOperator = Plus + Minus + Multiply + Divide + Modulate + Smaller + Greater;
	private static readonly string[] All =
	{
		Plus, Minus, Multiply, Divide, Modulate, Smaller, Greater, SmallerOrEqual, GreaterOrEqual,
		Is, And, Or, Xor, To
	};
	private static readonly string[] Arithmetic = { Plus, Minus, Multiply, Divide, Modulate };
	private static readonly string[] Comparison =
	{
		Is, Smaller, Greater, SmallerOrEqual, GreaterOrEqual
	};
	private static readonly string[] Logical = { And, Or, Xor };
	private static readonly string[] Conversions = { To };

	/// <summary>
	/// Example: 1+2*3%4 to Text is "1" becomes: ((1+(2*(3%4))) to Text) is "1"
	/// "5" to Number >= 6 is false becomes: (5 >= 6) is false
	/// </summary>
	/// <param name="tokenFirstCharacter"></param>
	/// <returns></returns>
	/// <exception cref="NotSupportedException"></exception>
	public static int GetPrecedence(char tokenFirstCharacter) =>
		tokenFirstCharacter switch
		{
			'+' => 1,
			'-' => 1,
			'*' => 2,
			'/' => 2,
			'^' => 3,
			'%' => 4,
			'<' => 6,
			'>' => 6,
			_ => throw new NotSupportedException(tokenFirstCharacter.ToString()) //ncrunch: no coverage
		};

	public static int GetPrecedence(ReadOnlySpan<char> token)
	{
		if (token.Compare(To))
			return 5;
		if (token.Compare(SmallerOrEqual) || token.Compare(GreaterOrEqual))
			return 6;
		if (token.Compare(And) || token.Compare(Or) || token.Compare(Xor))
			return 7;
		if (token.Compare(Is))
			return 10;
		throw new NotSupportedException(token.ToString()); //ncrunch: no coverage
	}
}