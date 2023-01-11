using System;

namespace Strict.Language.Expressions;

public sealed class Text : Value
{
	public Text(Context context, string value) : base(context.GetType(Base.Text), value) { }
	public override string ToString() => "\"" + Data + "\"";

	public override bool Equals(Expression? other) =>
		other is Value v && (string)Data == (string)v.Data;

	/// <summary>
	/// Only called for input that does not contain any spaces or if there are spaces (like "Hello" +
	/// " World") they are already split via <see cref="PhraseTokenizer"/> and thus is only a string.
	/// </summary>
	public static Expression? TryParse(Body body, ReadOnlySpan<char> input) =>
		input.Length >= 2 && input[0] == '"' && input[^1] == '"'
			? input.Length > Limit.TextCharacterCount + 2
				? throw new TextExceededMaximumCharacterLimitUseMultiLine(body, input.Length)
				: new Text(body.Method, input.Slice(1, input.Length - 2).ToString())
			: null;

	public sealed class TextExceededMaximumCharacterLimitUseMultiLine : ParsingFailed
	{
		public TextExceededMaximumCharacterLimitUseMultiLine(Body body, int characterCount) : base(body, "Line has text with characters count " + characterCount + " but allowed maximum limit is " + Limit.TextCharacterCount) { }
	}
}