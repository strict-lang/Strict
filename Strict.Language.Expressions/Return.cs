namespace Strict.Language.Expressions;

public sealed class Return : Expression
{
	public Return(Expression value) : base(value.ReturnType) => Value = value;
	public Expression Value { get; }
	public override int GetHashCode() => Value.GetHashCode();
	public override string ToString() => Keyword.Return + " " + Value;
	public override bool Equals(Expression? other) => other is Return a && Equals(Value, a.Value); //ncrunch: no coverage

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(Keyword.Return, StringComparison.Ordinal)
			? new Return(line.Length <= Keyword.Return.Length
				? throw new MissingExpression(body)
				: body.Method.ParseExpression(body, line[7..]))
			: null;

	public sealed class MissingExpression : ParsingFailed
	{
		public MissingExpression(Body body) : base(body) { }
	}
}