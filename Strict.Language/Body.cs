using System.Collections.Generic;
using System;
using System.Linq;
using System.Runtime.CompilerServices;
using static System.Formats.Asn1.AsnWriter;

[assembly: InternalsVisibleTo("Strict.Language.Expressions.Tests")]

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
	public Body(Method method, int tabs = 0, Body? parent = null) : base(method.ReturnType)
	{
		Method = method;
		Tabs = tabs;
		Parent = parent;
		parent?.children.Add(this);
	}

	public Method Method { get; }
	public int Tabs { get; }
	private Body? Parent { get; }
	public readonly List<Body> children = new();
	public Range LineRange { get; internal set; }
	public int ParsingLineNumber { get; set; }
	internal string CurrentLine => Method.lines[ParsingLineNumber];

	/// <summary>
	/// Called when actually needed and code needs to run, usually triggered by
	/// Method.GetBodyAndParseIfNeeded and child bodies inside. After parsing all expressions in this
	/// body, we will validate them all here. If there are multiple expressions, this body is
	/// returned, otherwise just a single expression is directly returned and the body is discarded.
	/// The last expression return type must match our (method or caller) return type.
	/// </summary>
	public Expression Parse()
	{
		//https://deltaengine.fogbugz.com/f/cases/25714/
		var expressions = new List<Expression>();
		for (ParsingLineNumber = LineRange.Start.Value; ParsingLineNumber < LineRange.End.Value;
			ParsingLineNumber++)
			expressions.Add(Method.ParseLine(this, CurrentLine));
		SetExpressions(expressions);
		return Expressions.Count == 1
			? Expressions[0]
			: this;
	}

	internal Body SetExpressions(IReadOnlyList<Expression> expressions)
	{
		Expressions = expressions;
		if (Expressions.Count == 0)
			throw new SpanExtensions.EmptyInputIsNotAllowed();
		ParsingLineNumber--;
		var isLastExpressionReturn = Expressions[^1].GetType().Name == Base.Return;
		var lastExpression = Expressions[^1];
		if ((isLastExpressionReturn || IsMethodReturn()) && Method.Name != Base.Run &&
			!ChildHasMatchingMethodReturnType(Parent == null
				? Method.ReturnType
				: Parent.ReturnType, lastExpression))
			throw new ChildBodyReturnTypeMustMatchMethod(this, lastExpression.ReturnType);
		return !isLastExpressionReturn
			? this
			: Parent != null
				? this
				: throw new ReturnAsLastExpressionIsNotNeeded(this);
	}

	private bool IsMethodReturn() =>
		ParsingLineNumber > 0 && ParsingLineNumber < LineRange.End.Value &&
		!CurrentLine.StartsWith("\t\t", StringComparison.Ordinal);

	public IReadOnlyList<Expression> Expressions { get; private set; } = Array.Empty<Expression>();

	/// <summary>
	/// We don't have access to the specific expressions here, so we need to do GetType().Name checks
	/// </summary>
	private static bool
		ChildHasMatchingMethodReturnType(Type parentType, Expression lastExpression) =>
		lastExpression.GetType().Name == Base.Assignment && parentType.Name == Base.None ||
		lastExpression.GetType().Name == Base.Error ||
		lastExpression.ReturnType.IsCompatible(parentType);

	public sealed class ChildBodyReturnTypeMustMatchMethod : ParsingFailed
	{
		public ChildBodyReturnTypeMustMatchMethod(Body body, Type childReturnType) : base(body,
			$"Child body return type: {childReturnType} is not matching with Parent return type:" + $" {
				(body.Parent == null
					? body.Method.ReturnType
					: body.Parent.ReturnType)
			} in method line: {
				body.ParsingLineNumber
			}") { }
	}

	public sealed class ReturnAsLastExpressionIsNotNeeded : ParsingFailed
	{
		public ReturnAsLastExpressionIsNotNeeded(Body body) : base(body) { }
	}

	/// <summary>
	/// Dictionaries are slow and eats up a lot of memory, only created when needed.
	/// </summary>
	private Dictionary<string, Expression>? variables;

	public Body AddVariable(string name, Expression value)
	{
		if (FindVariableValue(name.AsSpan()) != null)
			throw new ValueIsNotMutableAndCannotBeChanged(this, name);
		variables ??= new Dictionary<string, Expression>(StringComparer.Ordinal);
		variables.Add(name, value);
		return this;
	}

	public sealed class ValueIsNotMutableAndCannotBeChanged : ParsingFailed
	{
		public ValueIsNotMutableAndCannotBeChanged(Body body, string name) : base(body, name) { }
	}

	public void UpdateVariable(string name, Expression value)
	{
		var variable = FindVariableValue(name);
		if (variable == null)
			throw new IdentifierNotFound(this, name);
		if (variable.IsMutable)
			throw new ValueIsNotMutableAndCannotBeChanged(this, name);
		variables![name] = value;
	}

	public sealed class IdentifierNotFound : ParsingFailed
	{
		public IdentifierNotFound(Body body, string name) : base(body, name + ", Variables in scope: " + body.GetAllVariablesNames().ToWordList()) { }
	}

	private List<string> GetAllVariablesNames()
	{
		var allVariables = variables?.Keys.ToList() ?? new List<string>();
		if (Parent != null)
			allVariables.AddRange(Parent.GetAllVariablesNames());
		return allVariables;
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
	public string GetLine(int lineNumber) => Method.lines[lineNumber];

	public Body? FindCurrentChild()
	{
		// ReSharper disable once ForCanBeConvertedToForeach
		for (var index = 0; index < children.Count; index++)
		{
			var child = children[index];
			if (child.LineRange.Start.Value <= ParsingLineNumber)
				continue;
			if (child.LineRange.Start.Value > ParsingLineNumber + 1)
				break;
			ParsingLineNumber += child.LineRange.End.Value - child.LineRange.Start.Value;
			return child;
		}
		return null;
	}
}