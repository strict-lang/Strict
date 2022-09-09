using System;
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
	public const string Equal = "=";
	public const string Modulate = "%";
	public const string Smaller = "<";
	public const string Greater = ">";
	public const string SmallerOrEqual = "<=";
	public const string GreaterOrEqual = ">=";
	public const string Is = "is";
	public const string In = "in";
	public const string IsNot = "is not";
	public const string To = "to";
	public const string And = "and";
	public const string Or = "or";
	public const string Xor = "xor";

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsOperator(this ReadOnlySpan<char> name) =>
		name.Length == 1
			? IsSingleCharacterOperator(name[0])
			: IsMultiCharacterOperator(name);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsSingleCharacterOperator(this char tokenFirstCharacter) =>
		AnySingleCharacterOperator.Contains(tokenFirstCharacter);

	private const string AnySingleCharacterOperator = Plus + Minus + Multiply + Divide + Modulate + Smaller + Greater + Power + Equal;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsMultiCharacterOperator(this string name)
	{
		// ReSharper disable once ForCanBeConvertedToForeach
		for (var index = 0; index < MultiCharacterOperators.Length; index++)
			if (MultiCharacterOperators[index] == name)
				return true;
		return false;
	}

	private static readonly string[] MultiCharacterOperators =
	{
		SmallerOrEqual, GreaterOrEqual, Is, IsNot, In, And, Or, Xor, To
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsMultiCharacterOperator(this ReadOnlySpan<char> name)
	{
		if (name.Length <= 3)
			// ReSharper disable once ForCanBeConvertedToForeach
			for (var index = 0; index < MultiCharacterOperators.Length; index++)
				if (name.Compare(MultiCharacterOperators[index]))
					return true;
		return false;
	}

	private static readonly string[] All =
	{
		Plus, Minus, Multiply, Divide, Modulate, Smaller, Greater, SmallerOrEqual, GreaterOrEqual,
		Is, IsNot, In, And, Or, Xor, To, Equal
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
			',' => 0, // ncrunch: no coverage always has to flush everything out; ',' cannot be reached because this method is called only for operators
			'=' => 1, // ncrunch: no coverage always has to flush everything out; ',' cannot be reached because this method is called only for operators
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

	// ReSharper disable once MethodTooLong
	public static int GetPrecedence(ReadOnlySpan<char> token)
	{
		if (token.Compare("not")) //https://deltaengine.fogbugz.com/f/cases/25695/
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
		if (token.Compare(In))
			return 13;
		// ReSharper disable once ConvertIfStatementToReturnStatement
		if (token.Compare(IsNot))
			return 14;
		return GetPrecedence(token[0]);
	}
}