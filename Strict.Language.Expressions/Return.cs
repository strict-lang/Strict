using System;

namespace Strict.Language.Expressions;

public sealed class Return : Expression
{
	public Return(Expression value) : base(value.ReturnType) => Value = value;
	public Expression Value { get; }
	public override int GetHashCode() => Value.GetHashCode();
	public override string ToString() => "return " + Value;
	public override bool Equals(Expression? other) => other is Return a && Equals(Value, a.Value);

	public static Expression? TryParse(Method method, string line) =>
		line.StartsWith("return", StringComparison.Ordinal)
			? TryParseReturn(method, line)
			: null;

	private static Expression TryParseReturn(Method method, string line) =>
		new Return((line.Length < "return ".Length
			? null
			: method.TryParse(line["return ".Length..])) ?? throw new MissingExpression(line));

	public sealed class MissingExpression : Exception
	{
		public MissingExpression(string line) : base(line) { }
	}
}