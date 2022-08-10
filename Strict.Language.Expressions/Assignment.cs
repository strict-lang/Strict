using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Let assigns a variable in a method, often a fixed value that is optimized away.
/// </summary>
public sealed class Assignment : Expression
{
	public Assignment(Method inMethod, string name, Expression value) : base(value.ReturnType)
	{
		if (!name.IsWord())
			throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		inMethod.AddVariable(name, value);
		Name = name;
		Value = value;
	}

	public string Name { get; }
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
		var remainingPartSpan = line.Text.AsSpan(4);
		var parts = remainingPartSpan.Split();
		parts.MoveNext();
		var name = parts.Current.ToString();
		if (!parts.MoveNext() || !parts.MoveNext())
			throw new IncompleteLet(line);
		var startOfValueExpression = 4 + name.Length + 1 + 1 + 1;
		return new Assignment(line.Method, name, line.Method.ParseExpression(line, startOfValueExpression..));
	}

	public sealed class IncompleteLet : ParsingFailed
	{
		public IncompleteLet(Method.Line line) : base(line) { }
	}
}