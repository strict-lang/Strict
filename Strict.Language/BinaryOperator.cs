using System;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <summary>
/// https://strict.dev/docs/Keywords
/// </summary>
public static class BinaryOperator
{
	//TODO: could be changed to Enum again, then use this as byte in ShuntingYard.operators, much more efficient
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
	public const string And = "and";
	public const string Or = "or";
	public const string Xor = "xor";
	//TODO: remove
	public static bool IsOperator(this string name) => All.Contains(name);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsOperator(this ReadOnlySpan<char> name) =>
		name.Length == 1
			? IsSingleCharacterOperator(name[0])
			: IsMultiCharacterOperator(name);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsSingleCharacterOperator(this char tokenFirstCharacter) =>
		AnySingleCharacterOperator.Contains(tokenFirstCharacter);

	private const string AnySingleCharacterOperator = Plus + Minus + Multiply + Divide + Modulate + Smaller + Greater;

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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsMultiCharacterOperator(this ReadOnlySpan<char> name)
	{
		foreach (var checkOperator in MultiCharacterOperators)
			if (name.Compare(checkOperator))
				return true;
		return false;
	}

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
	public static int GetPrecedence(char tokenFirstCharacter) =>
		tokenFirstCharacter switch
		{
			',' => 0, // always has to flush everything out
			'+' => 2, // unary '-' and 'not' operators have precendence 1
			'-' => 2,
			'*' => 3,
			'/' => 3,
			'^' => 4,
			'%' => 5,
			'<' => 8,
			'>' => 8,
			_ => throw new NotSupportedException(tokenFirstCharacter.ToString()) //ncrunch: no coverage
		};

	public static int GetPrecedence(ReadOnlySpan<char> token)
	{
		if (token.Compare("not"))
			return 1;
		if (token.Compare(To))
			return 7;
		if (token.Compare(SmallerOrEqual) || token.Compare(GreaterOrEqual))
			return 8;
		if (token.Compare(And))
			return 9;
		if (token.Compare(Xor))
			return 10;
		if (token.Compare(Or))
			return 11;
		if (token.Compare(Is))
			return 12;
		return GetPrecedence(token[0]);
	}
}