using Strict.Language;
using Strict.Expressions;

namespace Strict.Compiler.Roslyn;

/// <summary>
/// Goes through all the possible <see cref="Strict.Language.Expression"/> classes.
/// </summary>
public abstract class ExpressionVisitor
{
	public IReadOnlyList<string> VisitBody(Expression expression) =>
		expression switch
		{
			Body body => VisitBody(body),
			If ifExpression => VisitIf(ifExpression),
			SelectorIf selectorIf => VisitSelectorIf(selectorIf),
			For forExpression => VisitFor(forExpression),
			_ => [Visit(expression) + ";"]
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

	protected IReadOnlyList<string> VisitSelectorIf(SelectorIf selectorIf)
	{
		var block = new List<string> { "switch (" + Visit(selectorIf.Selector) + ")", "{" };
		foreach (var @case in selectorIf.Cases)
			block.Add("\tcase " + Visit(@case.Pattern) + ": return " + Visit(@case.Then) + ";");
		if (selectorIf.OptionalElse != null)
			block.Add("\tdefault: return " + Visit(selectorIf.OptionalElse) + ";");
		block.Add("}");
		return block;
	}

	protected static IReadOnlyList<string> Indent(IEnumerable<string> lines) =>
		lines.Select(line => '\t' + line).ToArray();

	public string Visit(Expression expression) =>
		expression switch
		{
			Body => throw new UseVisitBody(expression),
			Declaration assignment => Visit(assignment),
			Binary binary => Visit(binary),
			Return returnExpression => Visit(returnExpression),
			Not not => not.ToString(),
			MethodCall call => Visit(call),
			MemberCall member => Visit(member),
			Value value => Visit(value),
			ListCall => expression.ToString().Replace('(', '[').Replace(')', ']'),
			_ => expression.ToString() //ncrunch: no coverage
		};

	public sealed class UseVisitBody(Expression expression) : Exception(expression.ToString());
	protected abstract string Visit(Declaration declaration);

	protected string Visit(Binary binary) =>
		Visit(binary.Instance!) + " " + GetBinaryOperator(binary.Method.Name) + " " + Visit(binary.Arguments[0]);

	protected abstract string GetBinaryOperator(string methodName);
	protected abstract string Visit(Return returnExpression);
	protected abstract string Visit(MethodCall methodCall);
	protected abstract string Visit(MemberCall memberCall);
	protected abstract string Visit(Value value);
	protected abstract IReadOnlyList<string> VisitFor(For forExpression);
}