using System;

namespace Strict.Language.Expressions;

public sealed class Return : Expression
{
	public Return(Expression value) : base(value.ReturnType) => Value = value;
	public Expression Value { get; }
	public override int GetHashCode() => Value.GetHashCode();
	public override string ToString() => "return " + Value;
	public override bool Equals(Expression? other) => other is Return a && Equals(Value, a.Value); //ncrunch: no coverage

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(ReturnName, StringComparison.Ordinal)
			? new Return(line.Length <= ReturnName.Length
				? throw new MissingExpression(body)
				: body.Method.ParseExpression(body, line[7..]))
			: null;

	private const string ReturnName = "return";

	public sealed class MissingExpression : ParsingFailed
	{
		public MissingExpression(Body body) : base(body) { }
	}
}