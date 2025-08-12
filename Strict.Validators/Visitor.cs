using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Validators;

/// <summary>
/// Visits any expression in types from a package, a type, a method and anything connected to
/// those (member initialization expressions, parameter default value expressions and of course
/// any body expressions). If a method body has not been parsed it won't be done here unless
/// explicitly stated, everything in Strict is lazily parsed. No one calls a method, no parsing!
/// If parsing happens later on, the BodyParsed event is triggered and the visitor is called
/// automatically here if it was used earlier on (e.g. by a Validator).
/// </summary>
public abstract class Visitor
{
	public void Visit(Package package, object? context = null)
	{
		foreach (var type in package)
			Visit(type, context);
	}

	public void Visit(Type type, object? context = null)
	{
		if (type.Name == Base.Any)
			return;
		foreach (var member in type.Members)
			TryVisit(member.InitialValue, context);
		foreach (var method in type.Methods)
			Visit(method, context: context);
	}

	public virtual void Visit(Method method, bool forceParsingBody = false, object? context = null)
	{
		foreach (var parameter in method.Parameters)
			TryVisit(parameter.DefaultValue, context);
		if (method.Type.IsTrait)
			return;
		if (forceParsingBody || method.WasParsedAlready)
			VisitBody(method.GetBodyAndParseIfNeeded(), context);
		else
			method.BodyParsed += body => VisitBody(body, context);
	}

	public virtual void VisitBody(Expression expression, object? context = null)
	{
		if (expression is Body body)
			foreach (var childExpression in body.Expressions)
				Visit(childExpression, context);
	}

	private void TryVisit(Expression? expression, object? context)
	{
		if (expression != null)
			Visit(expression, context);
	}

	public void Visit(Expression expression, object? context = null)
	{
		if (expression is Body body)
			VisitBody(body, context);
		else if (expression is Binary binary)
		{
			TryVisit(binary.Instance, context);
			Visit(binary.Arguments, context);
		}
		else if (expression is ConstantDeclaration declaration)
			// ReSharper disable TailRecursiveCall
			Visit(declaration.Value, context);
		else if (expression is MutableReassignment reassignment)
		{
			VisitExpression(reassignment, context);
			Visit(reassignment.Value, context);
		}
		else
			VisitExpression(expression, context);
	}

	private void Visit(IEnumerable<Expression> expressions, object? context)
	{
		foreach (var expression in expressions)
			Visit(expression, context);
	}

	protected abstract void VisitExpression(Expression expression, object? context);
}