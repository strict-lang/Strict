using System;

namespace Strict.Language.Expressions;

public sealed class Return : Expression
{
	public Return(Expression value) : base(value.ReturnType) => Value = value;
	public Expression Value { get; }
	public override int GetHashCode() => Value.GetHashCode();
	public override string ToString() => "return " + Value;
	public override bool Equals(Expression? other) => other is Return a && Equals(Value, a.Value);

	public static Expression? TryParse(Method.Line line) =>
		line.Text.StartsWith(ReturnName, StringComparison.Ordinal)
			? new Return(line.Text.Length <= ReturnName.Length
				? throw new MissingExpression(line)
				: line.Method.ParseExpression(line, (ReturnName.Length + 1)..))
			: null;

	private const string ReturnName = "return";

	public sealed class MissingExpression : ParsingFailed
	{
		public MissingExpression(Method.Line line) : base(line) { }
	}
}