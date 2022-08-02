using System;

namespace Strict.Language.Expressions;

public class Text : Value
{
	public Text(Context context, string value) : base(context.GetType(Base.Text), value) { }
	public override string ToString() => "\"" + Data + "\"";

	public override bool Equals(Expression? other) =>
		other is Value v && (string)Data == (string)v.Data;

	/// <summary>
	/// Only called for input that does not contain any spaces or if there are spaces (like "Hello" +
	/// " World") they are already split via <see cref="PhraseTokenizer"/> and thus is only a string.
	/// </summary>
	public static Expression? TryParse(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		return input.Length >= 2 && input[0] == '"' && input[^1] == '"'
			? new Text(line.Method, input.Slice(1, input.Length - 2).ToString())
			: null;
	}
}