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
		foreach (var member in type.Members)
			TryVisitExpression(member.InitialValue, context);
		foreach (var method in type.Methods)
			Visit(method, context: context);
	}

	public void Visit(Method method, bool forceParsingBody = false, object? context = null)
	{
		foreach (var parameter in method.Parameters)
			TryVisitExpression(parameter.DefaultValue, context);
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

	public void Visit(Expression expression, object? context = null)
	{
		if (expression is Body body)
			VisitBody(body, context);
		else
			VisitExpression(expression, context);
	}

	private void TryVisitExpression(Expression? expression, object? context)
	{
		if (expression != null)
			VisitExpression(expression, context);
	}

	protected abstract void VisitExpression(Expression expression, object? context);
}