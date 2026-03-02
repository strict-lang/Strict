using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Number(Context context, double value, int lineNumber = 0)
	: Value(context.GetType(Type.Number), value, lineNumber)
{
	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.TryParseNumber(out var number)
			? new Number(body.Method, number, body.CurrentFileLineNumber)
			: null;
}