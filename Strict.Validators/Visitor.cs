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
			Visit(member, context);
		foreach (var method in type.Methods)
			Visit(method, context: context);
	}

	protected virtual void Visit(Member member, object? context = null) =>
		member.InitialValue = Visit(member.InitialValue, null!, context);

	public virtual void Visit(Method method, bool forceParsingBody = false, object? context = null)
	{
		foreach (var parameter in method.Parameters)
			parameter.DefaultValue = Visit(parameter.DefaultValue, null!, context);
		if (method.Type.IsTrait)
			return;
		if (forceParsingBody || method.WasParsedAlready)
			TryVisitBody(method.GetBodyAndParseIfNeeded(), method, context);
		else
			method.BodyParsed += body => TryVisitBody(body, method, context);
	}

	private Expression TryVisitBody(Expression expression, Method method, object? context = null)
	{
		if (expression is Body body)
		{
			Visit(body, context);
			return body;
		}
		var replaced = Visit(expression, null, context)!;
		if (!ReferenceEquals(replaced, expression))
			method.SetBodySingleExpression(replaced);
		return replaced;
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
			var replaced = Visit(current, body, context);
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

	protected virtual Expression? Visit(Expression? expression, Body? body, object? context = null)
	{
		if (expression == null)
			return expression;
		if (expression is Body innerBody)
			Visit(innerBody, context);
		if (expression is Binary binary)
		{
			var changedInstance = Visit(binary.Instance, body, context)!;
			var rewrittenArgument = Visit(binary.Arguments[0], body, context)!;
			if (!ReferenceEquals(changedInstance, binary.Instance) ||
				!ReferenceEquals(rewrittenArgument, binary.Arguments[0]))
				return new Binary(changedInstance, binary.Method, [rewrittenArgument]);
		}
		if (expression is Declaration declaration)
		{
			var newValue = Visit(declaration.Value, body, context)!;
			if (!ReferenceEquals(newValue, declaration.Value) && body != null)
			{
				body.FindVariable(declaration.Name)!.InitialValue = newValue;
				declaration.SetValue(newValue);
			}
		}
		else if (expression is MutableReassignment reassignment)
			Visit(reassignment.Value, body, context);
		else if (expression is For forExpression)
		{
			Visit(forExpression.Value, body, context);
			Visit(forExpression.Body, body, context);
		}
		else if (expression is If ifExpression)
		{
			Visit(ifExpression.Condition, body, context);
			Visit(ifExpression.Then, body, context);
			Visit(ifExpression.OptionalElse, body, context);
		}
		else if (expression is ListCallStatement listCall)
		{
			Visit(listCall.List, body, context);
			Visit(listCall.Index, body, context);
		}
		else if (expression is MemberCall memberCall)
			Visit(memberCall.Instance, body, context);
		else if (expression is MethodCall methodCall)
		{
			Visit(methodCall.Instance, body, context);
			foreach (var argument in methodCall.Arguments)
				Visit(argument, body, context);
		}
		return expression;
	}
}