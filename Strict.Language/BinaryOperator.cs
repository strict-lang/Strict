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
	public const string To = "to";
	public const string And = "and";
	public const string Or = "or";
	public const string Xor = "xor";
	/*maybe should be avoided and parsing should handle this!
	/// <summary>
	/// These are combinations of the above operators that often happen, easier to parse directly.
	/// Method names are always single names (not and in here), the "is" is parsed, but not needed
	/// to find the method. The "is not in" just calls the in method and inverts the result.
	/// </summary>
	public const string IsNot = "is not";
	public const string IsIn = Is + " " + In;
	public const string IsNotIn = IsNot + " " + In;
	*/

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsOperator(this ReadOnlySpan<char> name) =>
		name.Length == 1
			? name[0].IsSingleCharacterOperator()
			: name.IsMultiCharacterOperator();

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsSingleCharacterOperator(this char tokenFirstCharacter) =>
		AnySingleCharacterOperator.Contains(tokenFirstCharacter);

	private const string AnySingleCharacterOperator =
		Plus + Minus + Multiply + Divide + Modulate + Smaller + Greater + Power;

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
		SmallerOrEqual, GreaterOrEqual, Is, In, And, Or, Xor, To, UnaryOperator.Not //, IsIn, IsNot, IsNotIn
	];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsMultiCharacterOperator(this ReadOnlySpan<char> name)
	{
		if (name.Length == 4 && name.Compare("from"))
			return true;
		if (name.Length is 2 or 3 or 5 or 6 or 9)
			// ReSharper disable once ForCanBeConvertedToForeach, not done for performance reasons
			for (var index = 0; index < MultiCharacterOperators.Length; index++)
				if (name.Compare(MultiCharacterOperators[index]))
					return true;
		return false;
	}

/*TODO
	extension(string input)
	{
		public bool IsMultiCharacterOperatorWithSpace(int currentIndex, out int tokenEnd)
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
		private bool IsNotInOperator(int currentIndex) =>
			input[currentIndex..].Length > 7 &&
			input[(currentIndex - 2)..(currentIndex + 8)] == IsNotIn + " ";

		private bool IsNotOperator(int currentIndex) =>
			input[currentIndex..].Length > 4 &&
			input[(currentIndex - 2)..(currentIndex + 5)] == IsNot + " ";

		private bool IsInOperator(int currentIndex) =>
			input[currentIndex..].Length > 3 &&
			input[(currentIndex - 2)..(currentIndex + 4)] == IsIn + " ";
	}
*/
	private static readonly string[] All =
	[
		Plus, Minus, Multiply, Divide, Modulate, Smaller, Greater, SmallerOrEqual, GreaterOrEqual,
		Is, In, And, Or, Xor, To
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
			',' => 0, // ncrunch: no coverage always has to flush everything out; ',' cannot be reached
             // because this method is called only for operators
			'+' => 11, // unary '-' and 'not' operators have precedence 10
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
								// "is not" and "is not in" will be flipped after we have all tokens
								: token.Compare(UnaryOperator.Not)
									? 3
									: token.Compare(Is)
										? 1
										: GetPrecedence(token[0]);
}