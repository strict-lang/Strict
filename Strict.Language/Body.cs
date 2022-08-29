using System.Collections.Generic;
using System;
using System.Linq;
using System.Runtime.CompilerServices;

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
	public Body(Method? method, int tabs = 0, Body? parent = null) : base(method?.ReturnType ?? new Type(new Package(null, ""), new TypeLines(""))) //TODO: Dummy initilization to avoid forced null
	{
		Method = method!;
		Tabs = tabs;
		Parent = parent;
		parent?.children.Add(this);
	}

	public Method Method { get; }
	public int Tabs { get; }
	public Body? Parent { get; }
	//TODO: make private again
	public readonly List<Body> children = new();
	public Range LineRange { get; internal set; }
	public int ParsingLineNumber { get; set; }
	public string CurrentLine => Method.lines[ParsingLineNumber];

	public void PushNestedBody(Body child) //TODO: what?
	{
		if (Expressions.Count == 0)
			Expressions = new List<Expression>();
		((List<Expression>)Expressions).Add(child);
	}

	/// <summary>
	/// Called when actually needed and code needs to run, usually triggered by
	/// Method.GetBodyAndParseIfNeeded and child bodies inside. After parsing all
	/// expressions in this body, we will validate them all here. If there are multiple expressions,
	/// this body is returned, otherwise just a single expression is directly returned and the body
	/// is discarded. The last expression return type must match our (method or caller) return type.
	/// </summary>
	public Expression Parse()
	{
		//TODO: we should probably split test expressions from production expressions here!
		var expressions = new List<Expression>();
		for (ParsingLineNumber = LineRange.Start.Value; ParsingLineNumber < LineRange.End.Value;
			ParsingLineNumber++)
			expressions.Add(Method.ParseLine(this));
		SetExpressions(expressions);
		return Expressions.Count == 1
			? Expressions[0]
			: this;
	}

	// ReSharper disable once MethodTooLong
	internal Body SetExpressions(IReadOnlyList<Expression> expressions)
	{
		Expressions = expressions;
		if (Expressions.Count == 0)
			throw new SpanExtensions.EmptyInputIsNotAllowed();
		var isLastExpressionReturn = Expressions[^1].GetType().Name == Base.Return;
		if (!isLastExpressionReturn)
			return this;
		var lastExpressionType = Expressions[^1].ReturnType;
		if (Parent != null)
			return isLastExpressionReturn &&
				!ChildHasMatchingMethodReturnType(Parent.ReturnType, lastExpressionType)
					? throw new ChildBodyReturnTypeMustMatchMethodReturnType(this, lastExpressionType)
					: this;
		ParsingLineNumber--;
		throw new ReturnAsLastExpressionIsNotNeeded(this);
	}

	public IReadOnlyList<Expression> Expressions { get; private set; } = Array.Empty<Expression>();

	private static bool ChildHasMatchingMethodReturnType(Type parentType, Type childType) =>
		childType.Name == Base.None || parentType == childType || childType.Implements.Contains(parentType);

	// ReSharper disable once HollowTypeName
	public class ChildBodyReturnTypeMustMatchMethodReturnType : ParsingFailed
	{
		public ChildBodyReturnTypeMustMatchMethodReturnType(Body body, Type childReturnType) : base(body, $"Child body return type: {childReturnType} is not matching with Parent return type: {body.Parent?.ReturnType} in method line: {--body.ParsingLineNumber}") { }
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