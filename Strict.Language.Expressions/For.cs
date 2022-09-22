using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses for loop expressions. Usually loop would have an implicit variable if not explicitly given anything,
/// so the variable is parsed with the first value of the iterable,
/// e.g for list the first element of the list or for range from 0
/// If explicit variable is given, the variable is added in the body, similarly to implicit index/value variables.
/// </summary>
public sealed class For : Expression
{
	private const string ForName = "for";
	private const string ValueName = "value";
	private const string IndexName = "index";
	private const string InName = "in";

	private For(Expression value, Expression body) : base(value.ReturnType)
	{
		Value = value;
		Body = body;
	}

	public Expression Value { get; }
	public Expression Body { get; }
	public override int GetHashCode() => Value.GetHashCode();
	public override string ToString() => $"for {Value}\n\t{Body}";
	public override bool Equals(Expression? other) => other is For a && Equals(Value, a.Value);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line)
	{
		if (!line.StartsWith(ForName, StringComparison.Ordinal))
			return null;
		if (line.Length <= ForName.Length)
			throw new MissingExpression(body);
		var innerBody = body.FindCurrentChild();
		if (innerBody == null)
			throw new MissingInnerBody(body);
		if (line.Contains(IndexName, StringComparison.Ordinal))
			throw new IndexIsReserved(body);
		return ParseFor(body, line, innerBody);
	}

	private static Expression ParseFor(Body body, ReadOnlySpan<char> line, Body innerBody)
	{
		if (!line.Contains(InName, StringComparison.Ordinal) ||
			line.Contains(IndexName, StringComparison.Ordinal))
			return ParseWithImplicitVariable(body, line, innerBody);
		CheckForUnidentifiedIterable(body, line);
		return ParseWithExplicitVariable(body, line, innerBody);
	}

	private static Expression ParseWithExplicitVariable(Body body,
		ReadOnlySpan<char> line, Body innerBody)
	{
		var variableName = FindVariableName(line);
		var variableValue = body.FindVariableValue(variableName);
		if (variableValue == null)
			body.AddVariable(variableName.ToString(),
				body.Method.ParseExpression(body, GetVariableExpressionValue(line)));
		if (body.FindVariableValue(variableName)?.ReturnType.Name != Base.Mutable)
			throw new ImmutableIterator(body);
		var forValueExpression = body.Method.ParseExpression(body, line[4..]);
		CheckForIncorrectMatchingTypes(body, variableName, forValueExpression);
		return new For(forValueExpression, innerBody.Parse());
	}

	private static void CheckForIncorrectMatchingTypes(Body body, ReadOnlySpan<char> variableName,
		Expression forValueExpression)
	{
		var mutableValue = body.FindVariableValue(variableName) as Mutable;
		var iteratorValue = ((Binary)forValueExpression).Arguments[0].ReturnType.Name;
		if ((iteratorValue != Base.Range || mutableValue?.DataReturnType.Name != Base.Number)
			&& iteratorValue != mutableValue?.DataReturnType.Name)
			throw new IteratorTypeDoesNotMatchWithIterable(body);
	}

	private static void CheckForUnidentifiedIterable(Body body, ReadOnlySpan<char> line)
	{
		if (body.FindVariableValue(FindIterableName(line)) == null && line[^1] != ')')
			throw new UnidentifiedIterable(body);
	}

	private static Expression ParseWithImplicitVariable(Body body, ReadOnlySpan<char> line,
		Body innerBody)
	{
		if (body.FindVariableValue(IndexName) != null)
			throw new DuplicateImplicitIndex(body);
		innerBody.AddVariable(IndexName, new Number(body.Method, 0));
		innerBody.AddVariable(ValueName,
			innerBody.Method.ParseExpression(innerBody, GetVariableExpressionValue(line)));
		return new For(body.Method.ParseExpression(body, line[4..]), innerBody.Parse());
	}

	private static string GetVariableExpressionValue(ReadOnlySpan<char> line) =>
		line.Contains("Range", StringComparison.Ordinal)
			? $"Mutable({GetRangeExpression(line)}.Start)"
			: $"Mutable({FindIterableName(line)}).First";

	private static ReadOnlySpan<char> GetRangeExpression(ReadOnlySpan<char> line) =>
		line[line.LastIndexOf('R')..(line.LastIndexOf(')') + 1)];

	private static ReadOnlySpan<char> FindVariableName(ReadOnlySpan<char> line) =>
		line[4..(line.LastIndexOf(InName) - 1)];

	private static ReadOnlySpan<char> FindIterableName(ReadOnlySpan<char> line) =>
		line.Contains(InName, StringComparison.Ordinal)
			? line[(line.LastIndexOf(InName) + 3)..]
			: line[(line.IndexOf(' ') + 1)..];

	public sealed class MissingExpression : ParsingFailed
	{
		public MissingExpression(Body body) : base(body) { }
	}

	public sealed class MissingInnerBody : ParsingFailed
	{
		public MissingInnerBody(Body body) : base(body) { }
	}

	public sealed class IndexIsReserved : ParsingFailed
	{
		public IndexIsReserved(Body body) : base(body) { }
	}

	public sealed class DuplicateImplicitIndex : ParsingFailed
	{
		public DuplicateImplicitIndex(Body body) : base(body) { }
	}

	public sealed class UnidentifiedIterable : ParsingFailed
	{
		public UnidentifiedIterable(Body body) : base(body) { }
	}

	public sealed class ImmutableIterator : ParsingFailed
	{
		public ImmutableIterator(Body body) : base(body) { }
	}

	public sealed class IteratorTypeDoesNotMatchWithIterable : ParsingFailed
	{
		public IteratorTypeDoesNotMatchWithIterable(Body body) : base(body) { }
	}
}