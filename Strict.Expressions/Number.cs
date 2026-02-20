using System.Globalization;
using Strict.Language;

namespace Strict.Expressions;

public sealed class Number(Context context, double value, int lineNumber = 0)
	: Value(context.GetType(Base.Number), value, lineNumber)
{
	public override string ToString() => ((double)Data).ToString(CultureInfo.InvariantCulture);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.TryParseNumber(out var number)
			? new Number(body.Method, number, body.CurrentFileLineNumber)
			: null;
}