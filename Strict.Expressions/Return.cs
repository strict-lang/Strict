using Strict.Language;

namespace Strict.Expressions;

public sealed class Return(Expression value, int lineNumber = 0)
	: Expression(value.ReturnType, lineNumber)
{
	public Expression Value { get; } = value;
	public override bool IsConstant => Value.IsConstant;
	public override int GetHashCode() => Value.GetHashCode();
	public override string ToString() => Keyword.Return + " " + Value;
	public override bool Equals(Expression? other) => other is Return a && Value.Equals(a.Value);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(Keyword.Return, StringComparison.Ordinal)
			? new Return(line.Length <= Keyword.Return.Length
					? throw new MissingExpression(body)
					: body.Method.ParseExpression(body, line[7..]),
				body.CurrentFileLineNumber)
			: null;

	public sealed class MissingExpression(Body body) : ParsingFailed(body);
}