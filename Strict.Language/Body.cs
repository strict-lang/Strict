using System.Collections.Generic;
using System;

namespace Strict.Language;

/// <summary>
/// Every method body is just an expression, which might contain multiple expressions, which are
/// all executed and then the final result is returned (all previous expressions must succeed).
/// Method parameters are in this context and can be used by any of the expressions nested here.
/// </summary>
public sealed class Body : Expression
{
	/// <summary>
	/// At construction time we only now the method we are in the if there is a parent Body we are in.
	/// While parsing each of the expressions we need to check for variables as defined below. This
	/// means the expressions list can't be done yet and needs this object to exist for scope parsing
	/// </summary>
	public Body(Method method, int tabs = 0 /*TODO: force this parameter to be used*/, Body? parent = null) : base(method.ReturnType)
	{
		Method = method;
		Tabs = tabs;
		Parent = parent;
	}

	public Method Method { get; }
	public int Tabs { get; }
	public Body? Parent { get; }

	public void PushNestedBody(Body child)
	{
		if (Expressions.Count == 0)
			Expressions = new List<Expression>();
		((List<Expression>)Expressions).Add(child);
	}

	/// <summary>
	/// After parsing each of the expressions in this body is done, we will validate them all here.
	/// In case this is the method body, the last expression return type must match our return type.
	/// </summary>
	public void SetAndValidateExpressions(IReadOnlyList<Expression> expressions,
		IReadOnlyList<Method.Line> bodyLines,
		//TODO: put these parameters first and force them!
		string Text = "", int FileLineNumber = 0)
	{
		if (expressions.Count == 0)
			throw new SpanExtensions.EmptyInputIsNotAllowed();
		if (expressions.Count == 1)
			throw new SingleExpressionBodyDoesNotMakeAnySense(bodyLines[0], expressions[0]);
		Expressions = expressions;
		if (Parent == null && expressions[^1].GetType().Name == "Return")
			throw new ReturnAsLastExpressionIsNotNeeded(bodyLines[^1]);
	}

	public sealed class SingleExpressionBodyDoesNotMakeAnySense : ParsingFailed
	{
		public SingleExpressionBodyDoesNotMakeAnySense(Method.Line line, Expression expression) :
			base(line, expression.ToString()) { }
	}

	public IReadOnlyList<Expression> Expressions { get; private set; } = Array.Empty<Expression>();

	public sealed class ReturnAsLastExpressionIsNotNeeded : ParsingFailed
	{
		public ReturnAsLastExpressionIsNotNeeded(Method.Line line) : base(line) { }
	}

	/// <summary>
	/// Dictionaries are slow and eats up a lot of memory, only created when needed.
	/// </summary>
	private Dictionary<string, Expression>? variables;

	public Body AddVariable(string name, Expression value)
	{
		variables ??= new Dictionary<string, Expression>(StringComparer.Ordinal);
		variables.Add(name, value);
		return this;
	}

	public Expression? FindVariableValue(ReadOnlySpan<char> searchFor)
	{
		if (variables != null)
			foreach (var (name, value) in variables)
				if (searchFor.Equals(name, StringComparison.Ordinal))
					return value;
		return Parent?.FindVariableValue(searchFor);
	}

	public override string ToString() => string.Join(Environment.NewLine, Expressions);
}