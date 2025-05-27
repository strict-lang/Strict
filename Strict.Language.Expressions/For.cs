namespace Strict.Language.Expressions;

/// <summary>
/// Parses for loop expressions. Usually loop would have an implicit variable if not explicitly given anything,
/// so the variable is parsed with the first value of the iterable,
/// e.g. for list the first element of the list or for range from 0
/// If explicit variable is given, the variable is added in the body, similarly to implicit index/value variables.
/// </summary>
public sealed class For : Expression
{
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
	private const string ValueName = "value";
	private const string IndexName = "index";
	private const string InName = "in ";
	private static bool HasIn(ReadOnlySpan<char> line) => line.Contains(InName, StringComparison.Ordinal);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line)
	{
		if (!line.StartsWith(Keyword.For, StringComparison.Ordinal))
			return null;
		if (line.Length <= Keyword.For.Length)
			throw new MissingExpression(body);
		var innerBody = body.FindCurrentChild() ??
			(TryGetInnerForAsBody(body) ?? throw new MissingInnerBody(body));
		if (line.Contains(IndexName, StringComparison.Ordinal))
			throw new IndexIsReserved(body);
		return ParseFor(body, line, innerBody);
	}

	private static Body? TryGetInnerForAsBody(Body body)
	{
		if (IsLastLine(body) || !IsNextLineStartsWithFor(body))
			return null;
		var currentLineNumber = body.ParsingLineNumber++;
		var child = body.FindCurrentChild();
		return child == null
			? null
			: body.GetInnerBodyAndUpdateHierarchy(currentLineNumber, child);
	}

	private static bool IsLastLine(Body body) => body.ParsingLineNumber + 1 == body.LineRange.End.Value;

	private static bool IsNextLineStartsWithFor(Body body) =>
		body.GetLine(body.ParsingLineNumber + 1).TrimStart().
			StartsWith(Keyword.For, StringComparison.Ordinal);

	public sealed class MissingInnerBody(Body body) : ParsingFailed(body);

	private static Expression ParseFor(Body body, ReadOnlySpan<char> line, Body innerBody)
	{
		if (HasIn(line) || line[^1] != ')')
		{
			var forExpression = ParseWithExplicitVariable(body, line, innerBody);
			if (!GetIteratorType(forExpression).IsIterator && forExpression.ReturnType.Name != Base.Number)
				throw new ExpressionTypeIsNotAnIterator(body, forExpression.ReturnType.Name,
					line[4..].ToString());
			return new For(forExpression, innerBody.Parse());
		}
		return ParseWithImplicitVariable(body, line, innerBody);
	}

	private static Expression ParseWithExplicitVariable(Body body,
		ReadOnlySpan<char> line, Body innerBody)
	{
		var variableName = FindIterableName(line);
		AddVariableIfDoesNotExist(body, line, variableName);
		if (body.FindVariableValue(variableName) is { IsMutable: false } && HasIn(line))
			throw new ImmutableIterator(body);
		var forExpression = body.Method.ParseExpression(body, GetForExpressionText(line));
		if (HasIn(line))
			CheckForIncorrectMatchingTypes(body, variableName, forExpression);
		else
			AddImplicitVariables(body, line, innerBody);
		return forExpression;
	}

	public sealed class ExpressionTypeIsNotAnIterator(Body body, string typeName, string line)
		: ParsingFailed(body, $"Type {typeName} in line " + line);

	private static void AddVariableIfDoesNotExist(Body body, ReadOnlySpan<char> line, ReadOnlySpan<char> variableName)
	{
		foreach (var variable in variableName.Split(',', StringSplitOptions.TrimEntries))
		{
			if (body.FindVariableValue(variable) != null)
				continue;
			var iterableName = variable.ToString();
			if (body.Method.Type.FindMember(iterableName) != null)
				continue;
			var variableValue = GetVariableExpression(body, line);
			variableValue.IsMutable = true;
			body.AddVariable(iterableName, variableValue);
		}
	}

	private static Expression GetVariableExpression(Body body, ReadOnlySpan<char> line)
	{
		var forIteratorText = GetForIteratorText(line);
		var iteratorExpression = body.Method.ParseExpression(body, forIteratorText);
		return iteratorExpression is MethodCall { ReturnType.Name: Base.Range } methodCall
			? GetVariableFromRange(iteratorExpression, methodCall)
			: iteratorExpression.ReturnType is GenericTypeImplementation { Generic.Name: Base.List }
				? body.Method.ParseExpression(body, forIteratorText[^1] == ')'
					? forIteratorText[1..forIteratorText.IndexOf(',')]
					: forIteratorText.ToString() + "(0)")
				: iteratorExpression;
	}

	private static Expression GetVariableFromRange(Expression iteratorExpression,
		MethodCall methodCall) =>
		methodCall.Arguments.Count > 0
			? methodCall.Arguments[0]
			: methodCall.Instance is MethodCall
			{
				ReturnType.Name: Base.Range, Arguments.Count: > 0
			} innerMethodCall
				? innerMethodCall.Arguments[0]
				: iteratorExpression;

	private static ReadOnlySpan<char> GetForIteratorText(ReadOnlySpan<char> line) =>
		line.Contains(InName, StringComparison.Ordinal)
			? line[(line.LastIndexOf(InName) + 3)..]
			: line.Contains("(", StringComparison.Ordinal) && line.IndexOf('(') > line.IndexOf(' ')
				? line[(line.IndexOf(' ') + 1)..]
				: line[(line.LastIndexOf(' ') + 1)..];

	private static ReadOnlySpan<char> GetForExpressionText(ReadOnlySpan<char> line) =>
		FindIterableName(line).Contains(',') && line.Contains("in", StringComparison.Ordinal)
			? line[(line.IndexOf(',') + 2)..]
			: line[4..];

	private static void CheckForIncorrectMatchingTypes(Body body, ReadOnlySpan<char> variableName,
		Expression forValueExpression)
	{
		foreach (var variable in variableName.Split(',', StringSplitOptions.TrimEntries))
		{
			var mutableValue = body.FindVariableValue(variable);
			var iteratorType = GetIteratorType(forValueExpression);
			if (iteratorType is GenericTypeImplementation { IsIterator: true } genericType)
				iteratorType = genericType.ImplementationTypes[0];
			if ((iteratorType.Name != Base.Range || mutableValue?.ReturnType.Name != Base.Number) &&
				iteratorType.Name != mutableValue?.ReturnType.Name)
				throw new IteratorTypeDoesNotMatchWithIterable(body, iteratorType.Name,
					mutableValue?.ReturnType.Name);
		}
	}

	private static Type GetIteratorType(Expression forValueExpression) =>
		forValueExpression is Binary binary
			? binary.Arguments[0].ReturnType
			: forValueExpression.ReturnType;

	private static Expression ParseWithImplicitVariable(Body body, ReadOnlySpan<char> line,
		Body innerBody)
	{
		if (body.FindVariableValue(IndexName) != null)
			throw new DuplicateImplicitIndex(body);
		AddImplicitVariables(body, line, innerBody);
		return new For(body.Method.ParseExpression(body, line[4..]), innerBody.Parse());
	}

	private static void AddImplicitVariables(Body body, ReadOnlySpan<char> line, Body innerBody)
	{
		innerBody.AddVariable(IndexName, new Number(body.Method, 0));
		var variableValue = innerBody.Method.ParseExpression(innerBody, GetVariableExpressionValue(body, line));
		variableValue.IsMutable = true;
		innerBody.AddVariable(ValueName, variableValue);
	}

	private static string GetVariableExpressionValue(Body body, ReadOnlySpan<char> line, ReadOnlySpan<char> knownIterableName = default)
	{
		if (line.Contains("Range", StringComparison.Ordinal))
			return $"{GetRangeExpression(line)}.Start";
		var iterableName = knownIterableName.IsEmpty
			? FindIterableName(line)
			: knownIterableName;
		var variable = body.FindVariableValue(iterableName)?.ReturnType ?? body.Method.Type.FindMember(iterableName.ToString())?.Type;
		return iterableName[^1] == ')'
			? iterableName[1..iterableName.IndexOf(',')].ToString()
			: variable is { IsIterator: true }
				? $"{iterableName}(0)"
				: $"{iterableName}";
	}

	private static ReadOnlySpan<char> GetRangeExpression(ReadOnlySpan<char> line) =>
		line[line.LastIndexOf("Range")..(line.LastIndexOf(')') + 1)];

	private static ReadOnlySpan<char> FindIterableName(ReadOnlySpan<char> line) =>
		line.Contains(InName, StringComparison.Ordinal)
			? line[4..(line.LastIndexOf(InName) - 1)]
			: line.Contains('.')
				? line[(line.IndexOf(' ') + 1)..line.IndexOf('.')]
				: line[(line.IndexOf(' ') + 1)..];

	public sealed class MissingExpression(Body body) : ParsingFailed(body);
	public sealed class IndexIsReserved(Body body) : ParsingFailed(body);
	public sealed class DuplicateImplicitIndex(Body body) : ParsingFailed(body);
	public sealed class ImmutableIterator(Body body) : ParsingFailed(body);

	public sealed class IteratorTypeDoesNotMatchWithIterable(Body body,
		string iteratorTypeName, string? variableType) : ParsingFailed(body,
		$"Iterator type {iteratorTypeName} does not match with {variableType}");
}