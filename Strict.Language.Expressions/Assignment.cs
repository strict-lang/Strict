using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Let assigns a variable in a method, often a fixed value that is optimized away.
/// </summary>
public sealed class Assignment : ConcreteExpression
{
	public Assignment(Body? scope, string name, Expression value) : base(value.ReturnType)
	{
		if (!name.IsWord())
			throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		var variable = scope?.FindVariableValue(name);
		if (variable != null && variable.ReturnType.IsMutable())
			scope?.UpdateVariable(name, value);
		else
			scope?.AddVariable(name, value);
		Name = name;
		Value = value;
	}

	public string Name { get; }
	public Expression Value { get; }
	public override int GetHashCode() => Name.GetHashCode() ^ Value.GetHashCode();
	public override string ToString() => Constant + " " + Name + " = " + Value;
	private const string Constant = "constant";

	public override bool Equals(Expression? other) =>
		other is Assignment a && Equals(Name, a.Name) && Value.Equals(a.Value);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(Constant + " ", StringComparison.Ordinal)
			? TryParseLet(body, line)
			: null;

	/// <summary>
	/// Highly optimized parsing of assignments, skips over the let, grabs the name of the local
	/// variable, then skips over the space, equal and space characters and parses the rest, e.g.
	/// constant hello = "hello" + " " + "world"
	///					 ^ ^       ^ ^   ^ ^       END, using TryParseExpression with Range(12, 35)
	/// </summary>
	private static Expression TryParseLet(Body body, ReadOnlySpan<char> line)
	{
		var parts = line[(Constant.Length + 1)..].Split();
		parts.MoveNext();
		var name = parts.Current.ToString();
		if (!parts.MoveNext() || !parts.MoveNext())
			throw new IncompleteLet(body);
		var startOfValueExpression = Constant.Length + 1 + name.Length + 1 + 1 + 1;
		return new Assignment(body, name, body.Method.ParseExpression(body, line[startOfValueExpression..]));
	}

	public sealed class IncompleteLet : ParsingFailed
	{
		public IncompleteLet(Body body) : base(body) { }
	}
}