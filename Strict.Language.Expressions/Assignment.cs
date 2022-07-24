using System;
using System.Collections.Generic;
using System.Linq;

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
		//let hello = "hello" + " " + "world"
		//         ^ ^       ^ ^   ^ ^       END
		//            Range(12, 35)
 		var parts = line.Text.AsSpan(4).Split();
		parts.MoveNext();
		var name = parts.Current.ToString();
		if (!parts.MoveNext() || !parts.MoveNext())
			throw new IncompleteLet(line);
		var startOfValueExpression = 4 + name.Length + 1 + 1 + 1;
		var value = line.Method.TryParseExpression(line, startOfValueExpression..) ??
			throw new MethodExpressionParser.UnknownExpression(line);
		return new Assignment(new Identifier(name, value.ReturnType), value);
	}

	public sealed class IncompleteLet : ParsingFailed
	{
		public IncompleteLet(Method.Line line) : base(line) { }
	}
}