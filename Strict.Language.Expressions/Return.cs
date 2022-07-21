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
			? TryParseReturn(line)
			: null;

	private const string ReturnName = "return ";

	private static Expression TryParseReturn(Method.Line line)
	{
		var returnExpression = line.Text.Length < ReturnName.Length
			? null
			: line.Method.TryParseExpression(line, line.Text.AsSpan(ReturnName.Length));
		return returnExpression == null
			? throw new MissingExpression(line)
			: new Return(returnExpression);
	}

	public sealed class MissingExpression : ParsingFailed
	{
		public MissingExpression(Method.Line line) : base(line) { }
	}
}