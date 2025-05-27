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
	public const string In = "in";
	public const string IsNot = "is not";
	public const string IsIn = Is + " " + In;
	public const string IsNotIn = IsNot + " " + In;
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

	private const string AnySingleCharacterOperator = Plus + Minus + Multiply + Divide + Modulate + Smaller + Greater + Power;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsMultiCharacterOperator(this string name)
	{
		// ReSharper disable once ForCanBeConvertedToForeach, not done for performance reasons
		for (var index = 0; index < MultiCharacterOperators.Length; index++)
			if (MultiCharacterOperators[index] == name)
				return true;
		return false;
	}

	private static readonly string[] MultiCharacterOperators =
	[
		SmallerOrEqual, GreaterOrEqual, Is, IsIn, IsNot, IsNotIn, In, And, Or, Xor, To,
		UnaryOperator.Not
	];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsMultiCharacterOperator(this ReadOnlySpan<char> name)
	{
		if (name.Length is 2 or 3 or 5 or 6 or 9)
			// ReSharper disable once ForCanBeConvertedToForeach, not done for performance reasons
			for (var index = 0; index < MultiCharacterOperators.Length; index++)
				if (name.Compare(MultiCharacterOperators[index]))
					return true;
		return false;
	}

	public static bool IsMultiCharacterOperatorWithSpace(this string input, int currentIndex, out int tokenEnd)
	{
		tokenEnd = 0;
		if (input[currentIndex - 1] != 's')
			return false;
		tokenEnd = input.IsNotInOperator(currentIndex)
			? 7
			: input.IsNotOperator(currentIndex)
				? 4
				: input.IsInOperator(currentIndex)
					? 3
					: 0;
		return tokenEnd is not 0;
	}

	private static bool IsNotInOperator(this string input, int currentIndex) =>
		input[currentIndex..].Length > 7 &&
		input[(currentIndex - 2)..(currentIndex + 8)] == IsNotIn + " ";

	private static bool IsNotOperator(this string input, int currentIndex) =>
		input[currentIndex..].Length > 4 &&
		input[(currentIndex - 2)..(currentIndex + 5)] == IsNot + " ";

	private static bool IsInOperator(this string input, int currentIndex) =>
		input[currentIndex..].Length > 3 &&
		input[(currentIndex - 2)..(currentIndex + 4)] == IsIn + " ";

	private static readonly string[] All =
	[
		Plus, Minus, Multiply, Divide, Modulate, Smaller, Greater, SmallerOrEqual, GreaterOrEqual,
		Is, IsNot, In, And, Or, Xor, To
	];
	private static readonly string[] Arithmetic = [Plus, Minus, Multiply, Divide, Modulate];
	private static readonly string[] Comparison =
	[
		Is, Smaller, Greater, SmallerOrEqual, GreaterOrEqual
	];
	private static readonly string[] Logical = [And, Or, Xor];
	private static readonly string[] Conversions = [To];

	/// <summary>
	/// Example: 1+2*3%4 to Text is "1" becomes: ((1+(2*(3%4))) to Text) is "1"
	/// "5" to Number >= 6 is false becomes: (5 >= 6) is false
	/// </summary>
	public static int GetPrecedence(char tokenFirstCharacter) =>
		tokenFirstCharacter switch
		{
			',' => 0, // ncrunch: no coverage always has to flush everything out; ',' cannot be reached because this method is called only for operators
			'+' => 11, // unary '-' and 'not' operators have precendence 10
			'-' => 11,
			'%' => 12,
			'*' => 13,
			'/' => 13,
			'^' => 14,
			'<' => 7,
			'>' => 7,
			_ => throw new NotSupportedException(tokenFirstCharacter.ToString()) //ncrunch: no coverage
		};

	public static int GetPrecedence(ReadOnlySpan<char> token) =>
		token.Compare(To)
			? 8
			: token.Compare(In)
				? 9
				: token.Compare(SmallerOrEqual) || token.Compare(GreaterOrEqual)
					? 7
					: token.Compare(And)
						? 6
						: token.Compare(Xor)
							? 5
							: token.Compare(Or)
								? 4
								: token.Compare(UnaryOperator.Not)
									? 3
									: token.Compare(Is) || token.Compare(IsIn) || token.Compare(IsNotIn) ||
									token.Compare(IsNot)
										? 1
										: GetPrecedence(token[0]);
}