﻿using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Language.Expressions.Tests")]

namespace Strict.Language;

/// <summary>
/// Every method body is just an expression, which might contain multiple expressions, which are
/// all executed, and then the final result is returned (all previous expressions must succeed).
/// Method parameters are in this context and can be used by any of the expressions nested here.
/// </summary>
public sealed class Body : Expression
{
	/// <summary>
	/// At construction time, we only know the method we are in, if there is a parent Body we are in.
	/// While parsing each of the expressions, we need to check for variables as defined below. This
	/// means the expressions list can't be done yet and needs this object to exist for scope parsing
	/// </summary>
	public Body(Method method, int tabs = 0, Body? parent = null) : base(method.ReturnType)
	{
		//tst: Console.WriteLine(method + " Body " + ParsingLineNumber);
		Method = method;
		Tabs = tabs;
		Parent = parent;
		parent?.children.Add(this);
	}

	public Method Method { get; private set; }
	public int Tabs { get; }
	public Body? Parent { get; private set; }
	public readonly List<Body> children = new();
	public Range LineRange { get; internal set; }
	public int ParsingLineNumber { get; set; }
	internal string CurrentLine => Method.lines[ParsingLineNumber];
	public bool IsFakeBodyForMemberInitialization => Method.Name == Type.EmptyBody;

	/// <summary>
	/// Called when actually needed, and code needs to run, usually triggered by
	/// Method.GetBodyAndParseIfNeeded and child bodies inside. After parsing all expressions in this
	/// body, we will validate them all here. If there are multiple expressions, this body is
	/// returned, otherwise just a single expression is directly returned and the body is discarded.
	/// The last expression return type must match our (method or caller) return type.
	/// </summary>
	public Expression Parse()
	{
		//tst: Console.WriteLine(Method + " Body.Parse " + LineRange);
		var expressions = new List<Expression>();
		for (ParsingLineNumber = LineRange.Start.Value; ParsingLineNumber < LineRange.End.Value;
			ParsingLineNumber++)
			try
			{
				expressions.Add(Method.ParseLine(this, CurrentLine));
			}
			catch (ParsingFailed)
			{
				throw;
			}
			catch (Exception ex)
			{
				throw new ParsingFailed((Type)Method.Parent, Method.TypeLineNumber + ParsingLineNumber,
					CurrentLine, ex);
			}
		SetExpressions(expressions);
		return Expressions.Count == 1
			? Expressions[0]
			: this;
	}

	public Body SetExpressions(IReadOnlyList<Expression> expressions)
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

	public IReadOnlyList<Expression> Expressions { get; private set; } = [];

	/// <summary>
	/// We don't have access to the specific expressions here, so we need to do GetType().Name checks
	/// </summary>
	private static bool
		ChildHasMatchingMethodReturnType(Type parentType, Expression lastExpression) =>
		lastExpression.GetType().Name == Base.ConstantDeclaration && parentType.Name == Base.None ||
		lastExpression.ReturnType.Name == Base.Error ||
		lastExpression.ReturnType.IsSameOrCanBeUsedAs(parentType);

	public sealed class ChildBodyReturnTypeMustMatchMethod(Body body, Type childReturnType)
		: ParsingFailed(body,
			$"Child body return type: {childReturnType} is not matching with Parent return type:" + $" {
				(body.Parent == null
					? body.Method.ReturnType
					: body.Parent.ReturnType)
			} in method line: {
				body.ParsingLineNumber
			}");

	public sealed class ReturnAsLastExpressionIsNotNeeded(Body body) : ParsingFailed(body);
	/// <summary>
	/// Dictionaries are slow and eats up a lot of memory, only created when needed.
	/// </summary>
	public Dictionary<string, Expression>? Variables { get; private set; }

	public Body AddVariable(string name, Expression value)
	{
		if (name.IsKeyword())
			throw new NamedType.CannotUseKeywordsAsName(name);
		if (!name.Length.IsWithinLimit())
			throw new NamedType.NameLengthIsNotWithinTheAllowedLimit(name);
		CheckForNameWithDifferentTypeUsage(name, value);
		if (FindVariableValue(name.AsSpan()) is not null)
			throw new ValueIsNotMutableAndCannotBeChanged(this, name);
		return InitializeOrUpdateVariable(name, value);
	}

	private void CheckForNameWithDifferentTypeUsage(string name, Expression value)
	{
		var nameType = value.ReturnType.TryGetType(name.MakeFirstLetterUppercase());
		if (nameType != null && nameType != value.ReturnType)
			throw new VariableNameCannotHaveDifferentTypeNameThanValue(this, name,
				value.ReturnType.Name);
	}

	private Body InitializeOrUpdateVariable(string name, Expression value)
	{
		Variables ??= new Dictionary<string, Expression>(StringComparer.Ordinal);
		Variables.Add(name, value);
		return this;
	}

	public class VariableNameCannotHaveDifferentTypeNameThanValue(Body body,
		string variableNameType, string valueType) : ParsingFailed(body,
		$"Variable name {variableNameType} " + $"denotes different type than its value type {
			valueType
		}. Prefer using a different name");

	public sealed class ValueIsNotMutableAndCannotBeChanged(Body body, string name)
		: ParsingFailed(body, name);

	public void UpdateVariable(string name, Expression value)
	{
		var variableScopeBody = FindVariableBody(name);
		var variable = variableScopeBody?.FindVariableValue(name) ??
			throw new IdentifierNotFound(this, name);
		if (!variable.IsMutable)
			throw new ValueIsNotMutableAndCannotBeChanged(this, name);
		if (variableScopeBody.Variables != null)
			variableScopeBody.Variables[name] = value;
	}

	public sealed class IdentifierNotFound(Body body, string name) : ParsingFailed(body,
		name + ", Variables in scope: " + body.GetAllVariablesNames().ToWordList());

	private List<string> GetAllVariablesNames()
	{
		var allVariables = Variables?.Keys.ToList() ?? new List<string>();
		if (Parent != null)
			allVariables.AddRange(Parent.GetAllVariablesNames());
		return allVariables;
	}

	public Expression? FindVariableValue(ReadOnlySpan<char> searchFor)
	{
		if (Variables != null)
			foreach (var (name, value) in Variables)
				if (searchFor.Equals(name, StringComparison.Ordinal))
					return value;
		return Parent?.FindVariableValue(searchFor);
	}

	private Body? FindVariableBody(ReadOnlySpan<char> searchFor)
	{
		if (Variables != null)
			foreach (var (name, _) in Variables)
				if (searchFor.Equals(name, StringComparison.Ordinal))
					return this;
		return Parent?.FindVariableBody(searchFor);
	}

	public override string ToString() => string.Join(Environment.NewLine, Expressions);
	public string GetLine(int lineNumber) => Method.lines[lineNumber];

	public Body? FindCurrentChild()
	{
		// ReSharper disable once ForCanBeConvertedToForeach, not done for performance reasons
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

	public void UpdateCurrentAndChildrenMethod(Method implementationMethod)
	{
		Method = implementationMethod;
		foreach (var child in children)
			child.UpdateCurrentAndChildrenMethod(implementationMethod);
	}

	public Body GetInnerBodyAndUpdateHierarchy(int currentLineNumber, Body child)
	{
		var innerForBody = new Body(Method, Tabs, this)
		{
			LineRange = new Range(currentLineNumber + 1, child.LineRange.End)
		};
		innerForBody.children.Add(child);
		child.Parent = innerForBody;
		children.Remove(child);
		return innerForBody;
	}
}