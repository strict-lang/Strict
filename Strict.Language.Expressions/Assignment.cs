using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Let statements in strict, which usually assigns a fixed value that is optimized away.
/// </summary>
public class Assignment : Expression
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

	private static Expression TryParseLet(Method.Line line)
	{
		var parts = line.Text.Split(new[] { "let ", " = " }, StringSplitOptions.RemoveEmptyEntries);
		if (parts.Length != 2)
			throw new IncompleteLet(line);
		var value = line.Method.TryParseExpression(line, parts[1]) ??
			throw new MethodExpressionParser.UnknownExpression(line);
		return new Assignment(new Identifier(parts[0], value.ReturnType), value);
	}

	public sealed class IncompleteLet : Method.ParsingError
	{
		public IncompleteLet(Method.Line line) : base(line) { }
	}
}