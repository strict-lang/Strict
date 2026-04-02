using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Parses for loop expressions. Usually loop would have an implicit variable if not explicitly
/// given anything, so the variable is parsed with the first value of the iterable.
/// E.g., for a list the first element of the list or for range from 0. If an explicit variable is
/// given, the variable is added in the body, similarly to implicit index/value variables.
/// </summary>
public sealed class For(Expression[] customVariables,
	Expression iterator,
	Expression body,
	int lineNumber,
	string shorthandOperator = "") : Expression(iterator.ReturnType, lineNumber)
{
	public Expression[] CustomVariables { get; } = customVariables;
	public Expression Iterator { get; } = iterator;
	public Expression Body { get; } = body;
	public string ShorthandOperator { get; } = shorthandOperator;
	public override int GetHashCode() => Iterator.GetHashCode();

	public override string ToString() =>
		$"for {InCustomVariables()}{Iterator}" + Environment.NewLine + (ShorthandOperator.Length == 0
			? IndentExpression(Body)
			: "\t" + ShorthandOperator + " " + Body);

	private string InCustomVariables() =>
		CustomVariables.Length > 0
			? string.Join(", ", CustomVariables) + " in "
			: "";

	public override bool IsConstant => Iterator.IsConstant && Body.IsConstant;

	public override bool Equals(Expression? other) =>
		other is For forExpression && Iterator.Equals(forExpression.Iterator) &&
		Body.Equals(forExpression.Body);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line)
	{
		if (!line.StartsWith(Keyword.For, StringComparison.Ordinal))
			return null;
		if (line.Length <= Keyword.For.Length)
			return ParseForImplicitIteratorOfThis(body);
		var innerBody = body.FindCurrentChild() ??
			TryGetInnerForAsBody(body) ?? throw new MissingInnerBody(body);
		return line.Contains(Type.IndexLowercase, StringComparison.Ordinal)
			? throw new IndexIsReservedDoNotUseItExplicitly(body)
			: ParseFor(body, line, innerBody);
	}

	private static Expression ParseForImplicitIteratorOfThis(Body body)
	{
		var innerBody = body.FindCurrentChild() ??
			TryGetInnerForAsBody(body) ?? throw new MissingInnerBody(body);
		if (body.FindVariable(Type.ValueLowercase.AsSpan(), false) == null)
			Instance.Parse(body, body.Method);
		var implicitIteratorName = body.Method.Type.IsText
			? "characters"
			: Type.ValueLowercase;
		AddImplicitVariables(body, $"{Keyword.For} {implicitIteratorName}".AsSpan(), innerBody);
		return new For([], new Instance(body.Method.Type, body.CurrentFileLineNumber),
			innerBody.Parse(), body.CurrentFileLineNumber);
	}

	public sealed class IndexIsReservedDoNotUseItExplicitly(Body body) : ParsingFailed(body);

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

	private static bool IsLastLine(Body body) =>
		body.ParsingLineNumber + 1 == body.LineRange.End.Value;

	private static bool IsNextLineStartsWithFor(Body body) =>
		body.GetLine(body.ParsingLineNumber + 1).TrimStart().
			StartsWith(Keyword.For, StringComparison.Ordinal);

	public sealed class MissingInnerBody(Body body) : ParsingFailed(body);

	private static Expression ParseFor(Body body, ReadOnlySpan<char> line, Body innerBody)
	{
		if (!HasIn(line) && line[^1] == ')')
			return ParseWithImplicitVariable(body, line, innerBody);
		var variableNames = FindVariableNames(line);
		var variables = AddVariablesIfTheyDoNotExistYet(body, line, variableNames, innerBody);
		if (body.FindVariable(variableNames) is { IsMutable: false } && HasIn(line))
			throw new ImmutableIterator(variableNames.ToString(), body);
		var iterator = body.Method.ParseExpression(body, HasIn(line)
			? GetForIteratorText(line)
			: GetForExpressionText(line));
		if (HasIn(line))
			CheckForIncorrectMatchingTypes(innerBody, variableNames, iterator);
		else
			AddImplicitVariables(body, line, innerBody);
		if (!GetIteratorType(iterator).IsIterator && !iterator.ReturnType.IsNumber)
			throw new ExpressionTypeIsNotAnIterator(body, iterator.ReturnType.Name,
				line[4..].ToString());
		var innerLines = body.Method.GetLinesAndStripTabs(innerBody.LineRange, body);
		var shorthandOperator = GetForBodyShorthandOperator(innerLines);
		var forExpression = new For(variables, iterator, innerBody.Parse(),
			body.CurrentFileLineNumber, shorthandOperator);
#if DEBUG
		var originalLines = line.ToString() + Environment.NewLine + innerLines.ToLines();
		var generatedLines = forExpression.ToString();
		var normalizedGenerated = generatedLines.Replace(Type.ValueLowercase + ".", string.Empty,
			StringComparison.Ordinal);
		var normalizedOriginal = originalLines.Replace(Type.ValueLowercase + ".", string.Empty,
			StringComparison.Ordinal);
		if (generatedLines != originalLines && normalizedGenerated != normalizedOriginal &&
			!innerLines.Any(loopLine => loopLine.TrimStart().
				StartsWith(BinaryOperator.To + " ", StringComparison.Ordinal)))
			throw new GeneratedForExpressionDoesNotMatchInputExactly(body, forExpression,
				originalLines);
#endif
		return forExpression;
	}

	private static string GetForBodyShorthandOperator(IReadOnlyList<string> innerLines)
	{
		if (innerLines.Count != 1)
			return string.Empty;
		var line = innerLines[0].TrimStart();
		var separatorIndex = line.IndexOf(' ');
		if (separatorIndex <= 0)
			return string.Empty;
		var candidate = line[..separatorIndex];
		return candidate.AsSpan().IsOperator()
			? candidate
			: string.Empty;
	}

	private static bool HasIn(ReadOnlySpan<char> line) =>
		line.Contains(InWithSpaces, StringComparison.Ordinal);

	private const string InWithSpaces = " in ";
#if DEBUG
	private sealed class
		GeneratedForExpressionDoesNotMatchInputExactly(Body body, Expression @for, string line)
		: ParsingFailed(body,
			"\n" + @for.ToString().Replace("\t", "  ") + "\nOriginal lines:\n" +
			line.Replace("\t", "  "));
#endif
	private static Expression ParseWithImplicitVariable(Body body, ReadOnlySpan<char> line,
		Body innerBody)
	{
		AddImplicitVariables(body, line, innerBody);
		return new For([], body.Method.ParseExpression(body, line[4..], true), innerBody.Parse(),
			body.CurrentFileLineNumber);
	}

	private static void AddImplicitVariables(Body body, ReadOnlySpan<char> line, Body innerBody)
	{
		if (innerBody.FindVariable(Type.IndexLowercase) != null &&
			innerBody.FindVariable(Type.ValueLowercase) != null)
			return;
		innerBody.AddVariable(Type.IndexLowercase, new Number(body.Method, 0), true);
		var iteratorYieldType = TryGetIteratorYieldType(body, GetForIteratorText(line));
		if (iteratorYieldType != null)
		{
			innerBody.AddVariable(Type.ValueLowercase,
				new Instance(iteratorYieldType, body.CurrentFileLineNumber), true);
			return;
		}
		var valueExpression = body.Method.ParseExpression(body,
			GetVariableExpressionValue(body, line), true);
		valueExpression = ConvertIteratorExpressionToYieldedValueIfNeeded(body, valueExpression);
		if (valueExpression.ReturnType.IsList)
		{
			var listElementType = valueExpression.ReturnType is GenericTypeImplementation listType
				? listType.ImplementationTypes[0]
				: valueExpression.ReturnType;
			valueExpression = listElementType.IsGeneric
				? new Instance(listElementType, body.CurrentFileLineNumber)
				: new ListCall(valueExpression, new Number(body.Method, 0));
		}
		innerBody.AddVariable(Type.ValueLowercase, valueExpression, true);
	}

	private static Type? TryGetIteratorYieldType(Body body, ReadOnlySpan<char> iteratorText)
	{
		var iteratorType = GetIteratorSourceType(body, iteratorText);
		var iteratorMethod =
			iteratorType?.Methods.FirstOrDefault(method => method.Name == Keyword.For);
		return iteratorMethod?.ReturnType.IsIterator == true &&
			iteratorMethod.ReturnType is GenericTypeImplementation generic
				? generic.ImplementationTypes[0]
				: null;
	}

	private static Type? GetIteratorSourceType(Body body, ReadOnlySpan<char> iteratorText)
	{
		var variableType = body.FindVariable(iteratorText)?.Type ??
			body.Method.Type.FindMember(iteratorText.ToString())?.Type;
		if (variableType != null)
			return variableType;
		var openingBracketIndex = iteratorText.IndexOf('(');
		return openingBracketIndex > 0
			? body.Method.FindType(iteratorText[..openingBracketIndex].ToString())
			: null;
	}

	private static Expression ConvertIteratorExpressionToYieldedValueIfNeeded(Body body,
		Expression valueExpression)
	{
		if (!valueExpression.ReturnType.IsIterator || valueExpression.ReturnType.IsList ||
			valueExpression.ReturnType.IsNumber)
			return valueExpression;
		var iteratorMethod = valueExpression.ReturnType.Methods.FirstOrDefault(method =>
			method.Name == Keyword.For);
		return iteratorMethod?.ReturnType.IsIterator == true &&
			iteratorMethod.ReturnType is GenericTypeImplementation generic
				? new Instance(generic.ImplementationTypes[0], body.CurrentFileLineNumber)
				: valueExpression;
	}

	private static string GetVariableExpressionValue(Body body, ReadOnlySpan<char> line,
		ReadOnlySpan<char> knownIterableName = default)
	{
		if (line.Contains("Range", StringComparison.Ordinal))
			return $"{GetRangeExpression(line)}.Start";
		var iterableName = knownIterableName.IsEmpty
			? GetForIteratorText(line)
			: knownIterableName;
		var variable = body.FindVariable(iterableName)?.Type ??
			body.Method.Type.FindMember(iterableName.ToString())?.Type;
		if (iterableName.Length > 0 && iterableName[0] == '(' && iterableName[^1] == ')')
			return iterableName[1..iterableName.IndexOf(',')].ToString();
		if (variable is { IsIterator: true })
		{
			var isGenericIterator = variable.IsGeneric ||
				variable is GenericTypeImplementation { ImplementationTypes.Count: > 0 } implementation &&
				implementation.ImplementationTypes[0].IsGeneric;
			return isGenericIterator
				? iterableName.ToString()
				: $"{iterableName}(0)";
		}
		return $"{iterableName}";
	}

	private static ReadOnlySpan<char> GetRangeExpression(ReadOnlySpan<char> line) =>
		line[line.LastIndexOf("Range")..(line.LastIndexOf(')') + 1)];

	private static ReadOnlySpan<char> FindVariableNames(ReadOnlySpan<char> line) =>
		line.Contains(InWithSpaces, StringComparison.Ordinal)
			? line[4..line.LastIndexOf(InWithSpaces)]
			: "";

	private static Expression[] AddVariablesIfTheyDoNotExistYet(Body body, ReadOnlySpan<char> line,
		ReadOnlySpan<char> variableNames, Body innerBody)
	{
		if (variableNames.IsEmpty)
			return [];
		var variables = new List<Expression>();
		var variableIndex = variableNames.Contains(',')
			? 0
			: -1;
		foreach (var variable in variableNames.Split(',', StringSplitOptions.TrimEntries))
		{
			var existingVariable = body.FindVariable(variable);
			if (existingVariable != null)
			{
				variables.Add(new VariableCall(existingVariable));
				continue;
			}
			var name = variable.ToString();
			if (body.Method.Type.FindMember(name) != null ||
				body.Method.Parameters.FirstOrDefault(parameter => parameter.Name == name) != null)
			{
				var instanceVariable = body.FindVariable(Type.ValueLowercase);
				variables.Add(new MemberCall(instanceVariable != null
					? new VariableCall(instanceVariable)
					: null, body.Method.Type.FindMember(name)!));
				continue;
			}
			innerBody.AddVariable(name, GetVariableValue(body, line, variableIndex), false);
			variables.Add(new VariableCall(innerBody.Variables!.Last(), body.CurrentFileLineNumber));
			if (variableIndex >= 0)
				variableIndex++;
		}
		return variables.ToArray();
	}

	private static Expression GetVariableValue(Body body, ReadOnlySpan<char> line,
		int variableIndex)
	{
		var forIteratorText = GetForIteratorText(line);
		var iteratorYieldType = TryGetIteratorYieldType(body, forIteratorText);
		if (iteratorYieldType != null)
			return variableIndex > 0
				? throw new MoreThanTwoVariablesAreNotSupportedYet(body)
				: new Instance(iteratorYieldType, body.CurrentFileLineNumber);
		var iterator = body.Method.ParseExpression(body, forIteratorText, true);
		if (iterator is MethodCall { ReturnType.Name: Type.Range } methodCall)
			return GetVariableValueFromRange(iterator, methodCall);
		if (iterator.ReturnType is GenericTypeImplementation { Generic.Name: Type.List })
		{
			var firstValue = body.Method.ParseExpression(body, forIteratorText.Length > 0 &&
				forIteratorText[0] == '(' && forIteratorText[^1] == ')'
					? forIteratorText[1..forIteratorText.IndexOf(',')]
					: forIteratorText.ToString() + "(0)", true);
			if (variableIndex <= 0)
				return firstValue;
			var innerFirstValue = body.Method.ParseExpression(body, firstValue + "(0)", true);
			return variableIndex > 1
				? throw new MoreThanTwoVariablesAreNotSupportedYet(body)
				: innerFirstValue;
		}
		var iteratorMethod = iterator.ReturnType.Methods.FirstOrDefault(method =>
			method.Name == Keyword.For);
		if (iteratorMethod?.ReturnType.IsIterator == true)
			return variableIndex > 0
				? throw new MoreThanTwoVariablesAreNotSupportedYet(body)
				: new Instance(iteratorMethod.ReturnType.GetFirstImplementation(),
					body.CurrentFileLineNumber);
		return iterator;
	}

	public sealed class MoreThanTwoVariablesAreNotSupportedYet(Body body)
		: ParsingFailed(body, "More than 2 for variables are not supported yet");

	private static Expression
		GetVariableValueFromRange(Expression iterator, MethodCall methodCall) =>
		methodCall.Arguments.Count > 0
			? methodCall.Arguments[0]
			: methodCall.Instance is MethodCall
			{
				ReturnType.Name: Type.Range, Arguments.Count: > 0
			} innerMethodCall
				? innerMethodCall.Arguments[0]
				: iterator;

	private static ReadOnlySpan<char> GetForIteratorText(ReadOnlySpan<char> line) =>
		line.Contains(InWithSpaces, StringComparison.Ordinal)
			? line[(line.LastIndexOf(InWithSpaces) + InWithSpaces.Length)..]
			: line.Contains("(", StringComparison.Ordinal) && line.IndexOf('(') > line.IndexOf(' ')
				? line[(line.IndexOf(' ') + 1)..]
				: line[(line.LastIndexOf(' ') + 1)..];

	private static ReadOnlySpan<char> GetForExpressionText(ReadOnlySpan<char> line) =>
		FindVariableNames(line).Contains(',') && line.Contains(InWithSpaces, StringComparison.Ordinal)
			? line[4..line.IndexOf(',')].ToString() +
			line[(line.IndexOf(InWithSpaces) - 1)..].ToString()
			: line[4..];

	private static void CheckForIncorrectMatchingTypes(Body innerBody,
		ReadOnlySpan<char> variableNames, Expression forValueExpression)
	{
		var implementationDepth = 1;
		foreach (var variable in variableNames.Split(',', StringSplitOptions.TrimEntries))
		{
			var mutableValue = innerBody.FindVariable(variable);
			if (mutableValue == null)
				throw new Body.IdentifierNotFound(innerBody, variable.ToString());
			var iteratorType = GetIteratorType(forValueExpression);
			for (var depth = 0; depth < implementationDepth; depth++)
				if (iteratorType is GenericTypeImplementation { IsIterator: true } genericType)
					iteratorType = genericType.ImplementationTypes[0];
			if ((iteratorType.Name != Type.Range || !mutableValue.Type.IsNumber) &&
				iteratorType.Name != mutableValue.Type.Name &&
				!iteratorType.IsSameOrCanBeUsedAs(mutableValue.Type, false))
				throw new IteratorTypeDoesNotMatchWithIterable(innerBody, iteratorType.Name, variable,
					mutableValue.Type.Name);
			implementationDepth++;
		}
	}

	private static Type GetIteratorType(Expression forValueExpression) =>
		forValueExpression is Binary binary
			? binary.Arguments[0].ReturnType
			: forValueExpression.ReturnType;

	public sealed class ExpressionTypeIsNotAnIterator(Body body, string typeName, string line)
		: ParsingFailed(body, $"Type {typeName} in line " + line);

	public sealed class ImmutableIterator(string iteratorVariableName, Body body)
		: ParsingFailed(body, iteratorVariableName);

	public sealed class IteratorTypeDoesNotMatchWithIterable(Body body,
		string iteratorTypeName,
		ReadOnlySpan<char> variable,
		string? variableType) : ParsingFailed(body,
		$"Iterator {variable} type {iteratorTypeName} does not match with {variableType}");
}