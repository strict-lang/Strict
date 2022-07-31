using System;
using System.Collections.Generic;
using System.Linq;
using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.Compiler.Roslyn;

/// <summary>
/// Goes through all of the possible <see cref="Strict.Language.Expression"/> classes.
/// </summary>
public abstract class ExpressionVisitor
{
	public IReadOnlyList<string> VisitBlock(Expression expression) =>
		expression switch
		{
			MethodBody methodBody => VisitBlock(methodBody),
			If ifExpression => VisitBlock(ifExpression),
			_ => new[] { Visit(expression) + ";" }
		};

	protected abstract IReadOnlyList<string> VisitBlock(MethodBody methodBody);

	protected IReadOnlyList<string> VisitBlock(If ifExpression)
	{
		var block = new List<string> { "if (" + Visit(ifExpression.Condition) + ")" };
		block.AddRange(Indent(VisitBlock(ifExpression.Then)));
		if (ifExpression.OptionalElse == null)
			return block;
		block.Add("else");
		block.AddRange(Indent(VisitBlock(ifExpression.OptionalElse)));
		return block;
	}

	protected static IReadOnlyList<string> Indent(IEnumerable<string> lines) =>
		lines.Select(line => '\t' + line).ToArray();

	public string Visit(Expression expression) =>
		expression switch
		{
			BlockExpression => throw new UseVisitBlock(expression),
			Assignment assignment => Visit(assignment),
			Binary binary => Visit(binary),
			Return returnExpression => Visit(returnExpression),
			ArgumentsMethodCall call => Visit(call),
			OneArgumentMethodCall call => Visit(call),
			NoArgumentMethodCall call => Visit(call),
			MemberCall member => Visit(member),
			Value value => Visit(value),
			_ => expression.ToString() //ncrunch: no coverage
		};

	public sealed class UseVisitBlock : Exception
	{
		public UseVisitBlock(Expression expression) : base(expression.ToString()) { }
	}

	protected abstract string Visit(Assignment assignment);

	protected string Visit(Binary binary) =>
		Visit(binary.Instance!) + " " + GetBinaryOperator(binary.Method.Name) + " " + Visit(binary.Argument);

	protected abstract string GetBinaryOperator(string methodName);
	protected abstract string Visit(Return returnExpression);
	protected abstract string Visit(ArgumentsMethodCall methodCall);
	protected abstract string Visit(OneArgumentMethodCall methodCall);
	protected abstract string Visit(NoArgumentMethodCall methodCall);
	protected abstract string Visit(MemberCall memberCall);
	protected abstract string Visit(Value value);
}