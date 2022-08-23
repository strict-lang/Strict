using System;

namespace Strict.Language.Expressions;

//TODO: docusaurus link for all expressions!
public sealed class For : Expression
{
	public For(Expression value) : base(value.ReturnType) => Value = value;
	public Expression Value { get; } //TODO: for Range, for list, etc.

	//TODO: public Expression Body { get; }
	public override int GetHashCode() => Value.GetHashCode();
	public override string ToString() => $"for {Value}";
	public override bool Equals(Expression? other) => other is For a && Equals(Value, a.Value);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(ForName, StringComparison.Ordinal)
			? new For(line.Length <= ForName.Length
				? throw new MissingExpression(body)
				: body.Method.ParseExpression(body, line[4..]))
			: null;

	private const string ForName = "for";

	public sealed class MissingExpression : ParsingFailed
	{
		public MissingExpression(Body body) : base(body) { }
	}
}