using System.Globalization;

namespace Strict.Language.Expressions;

public sealed class Number : Value
{
	public Number(Context context, double value) : base(context.GetType(Base.Number), value) { }
	public override string ToString() => ((double)Data).ToString(CultureInfo.InvariantCulture);

	public override bool Equals(Expression? other) =>
		other is Value v && (double)Data == (double)v.Data;

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.TryParseNumber(out var number)
			? new Number(body.Method, number)
			: null;
}