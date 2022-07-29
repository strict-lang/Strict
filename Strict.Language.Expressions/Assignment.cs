using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Let assigns an <see cref="Identifier"/> variable, often a fixed value that is optimized away
/// </summary>
public sealed class Assignment : Expression
{
	public Assignment(Identifier name, Expression value) : base(value.ReturnType)
	{
		Name = name;
		Value = value;
	}

	public Identifier Name { get; }
	public Expression Value { get; }
	public override int GetHashCode() => Name.GetHashCode() ^ Value.GetHashCode();
	public override string ToString() => "let " + Name + " = " + Value;

	public override bool Equals(Expression? other) =>
		other is Assignment a && Equals(Name, a.Name) && Value.Equals(a.Value);

	public static Expression? TryParse(Method.Line line) =>
		line.Text.StartsWith("let ", StringComparison.Ordinal)
			? TryParseLet(line)
			: null;

	/// <summary>
	/// Highly optimized parsing of assignments, skips over the let, grabs the name of the local
	/// variable, then skips over the space, equal and space characters and parses the rest, e.g.
	/// let hello = "hello" + " " + "world"
	///          ^ ^       ^ ^   ^ ^       END, using TryParseExpression with Range(12, 35)
	/// </summary>
	private static Expression TryParseLet(Method.Line line)
	{
		var parts = line.Text.AsSpan(4).Split();
		parts.MoveNext();
		var name = parts.Current.ToString();
		if (!parts.MoveNext() || !parts.MoveNext())
			throw new IncompleteLet(line);
		var startOfValueExpression = 4 + name.Length + 1 + 1 + 1;
		var value = line.Method.ParseExpression(line, startOfValueExpression..);
		return new Assignment(new Identifier(name, value.ReturnType), value);
	}

	public sealed class IncompleteLet : ParsingFailed
	{
		public IncompleteLet(Method.Line line) : base(line) { }
	}
}