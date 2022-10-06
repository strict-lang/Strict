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
	public IReadOnlyList<string> VisitBody(Expression expression) =>
		expression switch
		{
			Body body => VisitBody(body),
			If ifExpression => VisitIf(ifExpression),
			_ => new[] { Visit(expression) + ";" }
		};

	protected abstract IReadOnlyList<string> VisitBody(Body body);

	protected IReadOnlyList<string> VisitIf(If ifExpression)
	{
		var block = new List<string> { "if (" + Visit(ifExpression.Condition) + ")" };
		block.AddRange(Indent(VisitBody(ifExpression.Then)));
		if (ifExpression.OptionalElse == null)
			return block;
		block.Add("else");
		block.AddRange(Indent(VisitBody(ifExpression.OptionalElse)));
		return block;
	}

	protected static IReadOnlyList<string> Indent(IEnumerable<string> lines) =>
		lines.Select(line => '\t' + line).ToArray();

	public string Visit(Expression expression) =>
		expression switch
		{
			Body => throw new UseVisitBody(expression),
			Assignment assignment => Visit(assignment),
			Binary binary => Visit(binary),
			Return returnExpression => Visit(returnExpression),
			MethodCall call => Visit(call),
			MemberCall member => Visit(member),
			Value value => Visit(value),
			ListCall => expression.ToString().Replace('(', '[').Replace(')', ']'),
			_ => expression.ToString() //ncrunch: no coverage
		};

	public sealed class UseVisitBody : Exception
	{
		public UseVisitBody(Expression expression) : base(expression.ToString()) { }
	}

	protected abstract string Visit(Assignment assignment);

	protected string Visit(Binary binary) =>
		Visit(binary.Instance!) + " " + GetBinaryOperator(binary.Method.Name) + " " + Visit(binary.Arguments[0]);

	protected abstract string GetBinaryOperator(string methodName);
	protected abstract string Visit(Return returnExpression);
	protected abstract string Visit(MethodCall methodCall);
	protected abstract string Visit(MemberCall memberCall);
	protected abstract string Visit(Value value);
}