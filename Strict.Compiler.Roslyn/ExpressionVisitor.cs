using System;
using Strict.Language;
using Strict.Language.Expressions;
using Boolean = Strict.Language.Expressions.Boolean;

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
			Binary binary => Visit(binary),
			Boolean boolean => Visit(boolean),
			MemberCall memberCall => Visit(memberCall),
			MethodCall methodCall => Visit(methodCall),
			Number number => Visit(number),
			Text text => Visit(text),
			Value value => Visit(value),
			_ => throw new ExpressionNotSupported(expression)
		};

	protected abstract string Visit(MethodBody methodBody);
	protected abstract string Visit(Assignment assignment);
	protected abstract string Visit(Binary binary);
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
}