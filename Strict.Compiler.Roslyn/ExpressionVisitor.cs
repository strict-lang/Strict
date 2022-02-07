using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.Compiler.Roslyn;

/// <summary>
/// Goes through all of the possible <see cref="Strict.Language.Expression"/> classes.
/// </summary>
public abstract class ExpressionVisitor
{
	public string Visit(Expression expression, int tabIndentation = 2) =>
		expression switch
		{
			MethodBody methodBody => Visit(methodBody),
			Assignment assignment => Visit(assignment),
			_ => VisitWhenExpressionIsSameInCSharp(expression)
		};

	protected abstract string Visit(MethodBody methodBody);
	protected abstract string Visit(Assignment assignment);
	protected abstract string VisitWhenExpressionIsSameInCSharp(Expression expression);
	/*
	protected abstract string Visit(Boolean boolean);
	protected abstract string Visit(MemberCall memberCall);
	protected abstract string Visit(MethodCall methodCall);
	protected abstract string Visit(Number number);
	protected abstract string Visit(Text text);
	protected abstract string Visit(Value value);

	public class ExpressionNotSupported : Exception
	{
		public ExpressionNotSupported(Expression expression) : base(expression.GetType().Name + ": " +
			expression) { }
	}
	*/
}