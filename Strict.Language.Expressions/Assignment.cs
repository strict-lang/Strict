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

	public override string ToString() =>
		(Value.ReturnType.IsMutable()
			? Mutable
			: Constant) + " " + Name + " = " + Value;

	private const string Mutable = "mutable";
	private const string Constant = "constant";

	public override bool Equals(Expression? other) =>
		other is Assignment a && Equals(Name, a.Name) && Value.Equals(a.Value);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(Constant + " ", StringComparison.Ordinal)
			? TryParseAssignment(body, line, Constant.Length + 1)
			: line.StartsWith(Mutable + " ", StringComparison.Ordinal)
				? TryParseAssignment(body, line, Mutable.Length + 1)
				: null;

	/// <summary>
	/// Highly optimized parsing of assignments, skips over the let, grabs the name of the local
	/// variable, then skips over the space, equal and space characters and parses the rest, e.g.
	/// constant hello = "hello" + " " + "world"
	///					 ^ ^       ^ ^   ^ ^       END, using TryParseExpression with Range(12, 35)
	/// </summary>
	private static Expression TryParseAssignment(Body body, ReadOnlySpan<char> line, int startIndex)
	{
		var parts = line[startIndex..].Split();
		parts.MoveNext();
		var name = parts.Current.ToString();
		if (!parts.MoveNext() || !parts.MoveNext())
			throw new MissingAssignmentValueExpression(body);
		var expressionSpan = line[(startIndex + name.Length + 1 + 1 + 1)..];
		var expression = expressionSpan.IsFirstLetterUppercase() && expressionSpan.IsPlural()
			? new List(body.Method.Type.GetType(expressionSpan.ToString()))
			: ParseExpressionAndCheckType(body, expressionSpan, name);
		return new Assignment(body, name, startIndex == 8
			? new Mutable(body.Method, expression)
			: expression);
	}

	private static Expression ParseExpressionAndCheckType(Body body,
		ReadOnlySpan<char> expressionSpan, ReadOnlySpan<char> variableName)
	{
		var expression = body.Method.ParseExpression(body, expressionSpan);
		if (expression.ReturnType.IsMutable())
			throw new DirectUsageOfMutableTypesOrImplementsAreForbidden(body, expressionSpan.ToString(),
				variableName.ToString());
		return expression;
	}

	public sealed class DirectUsageOfMutableTypesOrImplementsAreForbidden : ParsingFailed
	{
		public DirectUsageOfMutableTypesOrImplementsAreForbidden(Body body, string expressionText, string variableName) : base(body, $"Direct usage of mutable types or type that implements Mutable {expressionText} are not allowed. Instead use immutable types for variable {variableName}") { }
	}

	public sealed class MissingAssignmentValueExpression : ParsingFailed
	{
		public MissingAssignmentValueExpression(Body body) : base(body) { }
	}
}