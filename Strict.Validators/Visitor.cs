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

	public virtual void Visit(Type type, object? context = null)
	{
		if (type.Name == Base.Any)
			return;
		foreach (var member in type.Members)
			if (member.InitialValue != null)
			{
				var updatedExpression = Visit(member.InitialValue, context);
				//TODO: if (upda)
			}
		foreach (var method in type.Methods)
			Visit(method, context: context);
	}

	public virtual void Visit(Method method, bool forceParsingBody = false, object? context = null)
	{
		foreach (var parameter in method.Parameters)
			Visit(parameter.DefaultValue, context);
		if (method.Type.IsTrait)
			return;
		if (forceParsingBody || method.WasParsedAlready)
			TryVisitBody(method.GetBodyAndParseIfNeeded(), method, context);
		else
			method.BodyParsed += body => TryVisitBody(body, method, context);
	}

	private void TryVisitBody(Expression expression, Method method, object? context = null)
	{
		if (expression is Body body)
			Visit(body, context);
		else
		{
			var replaced = VisitExpression(expression, context);
			if (!ReferenceEquals(replaced, expression))
				method.SetBodySingleExpression(replaced);
		}
	}

	/// <summary>
	/// Rewrite body expressions only when needed; avoid allocations if nothing changed.
	/// </summary>
	protected virtual void Visit(Body body, object? context = null)
	{
		List<Expression>? rewritten = null;
		for (var i = 0; i < body.Expressions.Count; i++)
		{
			var current = body.Expressions[i];
			var replaced = Visit(current, context);
			// If an expression changed, create rewritten and copy previous untouched elements
			if (!ReferenceEquals(replaced, current))
			{
				rewritten ??= new List<Expression>(body.Expressions.Count);
				if (rewritten.Count == 0)
					for (var j = 0; j < i; j++)
						rewritten.Add(body.Expressions[j]);
				rewritten.Add(replaced!);
			}
			else
				rewritten?.Add(current);
		}
		if (rewritten != null)
			body.SetExpressions(rewritten);
	}

	protected virtual Expression? Visit(Expression? expression, object? context = null)
	{
		if (expression == null)
			return expression;
		if (expression is Body body)
			Visit(body, context);
		else if (expression is Binary binary)
		{
			Visit(binary.Instance, context);
			Visit(binary.Arguments, context);
		}
		else if (expression is ConstantDeclaration declaration)
			// ReSharper disable TailRecursiveCall
			Visit(declaration.Value, context);
		else if (expression is MutableReassignment reassignment)
		{
			Visit(reassignment.Value, context);
			return VisitExpression(reassignment, context);
		}
		else
			return VisitExpression(expression, context);
		return expression;
	}

	private void Visit(IEnumerable<Expression> expressions, object? context)
	{
		foreach (var expression in expressions)
			Visit(expression, context);
	}

	protected abstract Expression VisitExpression(Expression expression, object? context);
}