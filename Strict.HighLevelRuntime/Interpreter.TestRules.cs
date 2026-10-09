using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict.HighLevelRuntime.Tests")]
[assembly: InternalsVisibleTo("Strict.TestRunner")]

namespace Strict.HighLevelRuntime;

public partial class Interpreter
{
	private static bool ShouldIgnoreGenericListTestParseFailure(Method method, Exception inner) =>
		method.Type.IsGeneric && method.Type.Name == Type.List &&
		inner is Type.GenericTypesCannotBeUsedDirectlyUseImplementation;

	private static bool IsKnownParserLimitation(Exception inner) =>
		inner is ParsingFailed &&
		(inner.InnerException is Type.NoMatchingMethodFound
				or Type.ArgumentsDoNotMatchMethodParameters ||
			inner.Message.Contains("Use number iteration"));

	private static bool ShouldSkipGenericListTestValidation(Method method, bool runOnlyTests) =>
		runOnlyTests && method.Type is { IsGeneric: true, Name: Type.List or Type.Dictionary };

	private static bool ShouldSkipKnownStrictBaseMethodValidation(Method method, bool runOnlyTests) =>
		runOnlyTests && ((method.Type.IsGeneric && method.Type.Name == Type.List) ||
			(method.Type.Name == Type.Number && (method.Name == "digits" ||
				(method.Name == BinaryOperator.To && method.ReturnType.IsText))) ||
			(method.Type.IsText && method.Name == "Split") ||
			method.Type.Name is "Parser" or "ShuntingYard");

	/// <summary>
	/// Skip parsing for trivially simple methods during validation to avoid missing-instance errors.
	/// </summary>
	private bool IsSimpleSingleLineMethod(Method method) =>
		simpleMethodCache.GetOrAdd(method, CheckIsSimpleSingleLineMethod);

	private static bool CheckIsSimpleSingleLineMethod(Method method)
	{
		if (method.lines.Count != 2)
			return false;
		var bodyLine = method.lines[1].Trim();
		var hasMethodCalls = bodyLine.Contains('(') && !bodyLine.StartsWith('(');
		if (hasMethodCalls)
			return false;
		var thenCount = CountThenSeparators(bodyLine);
		var operatorCount = CountOperatorWords(bodyLine);
		return (thenCount == 0 && operatorCount <= 1) || (thenCount == 1 && operatorCount <= 2) ||
			(thenCount == 2 && operatorCount == 0);
	}

	private static int CountOperatorWords(string input)
	{
		var span = input.AsSpan();
		var count = 0;
		while (span.Length > 0)
		{
			var spaceIndex = span.IndexOf(' ');
			var word = spaceIndex < 0
				? span
				: span[..spaceIndex];
			if (word is "and" or "or" or "not" or "is")
				count++;
			if (spaceIndex < 0)
				break;
			span = span[(spaceIndex + 1)..];
		}
		return count;
	}

	private static int CountThenSeparators(string input)
	{
		var count = 0;
		for (var index = 0; index <= input.Length - If.ThenSeparator.Length; index++)
			if (input.AsSpan(index).StartsWith(If.ThenSeparator, StringComparison.Ordinal))
			{
				count++;
				index += If.ThenSeparator.Length - 1;
			}
		return count;
	}

	/// <summary>
	/// Simple expressions like "value is other" or "value" don't need tests as they are
	/// essentially getters or trivial delegations. Complex expressions need tests.
	/// </summary>
	private static bool IsSimpleExpressionWithLessThanThreeSubExpressions(Expression expr) =>
		CountExpressionComplexity(expr) <= MaxSimpleExpressionComplexity;

	private static int CountExpressionComplexity(Expression expr) =>
		expr switch
		{
			Binary => 1,
			Not n => 1 + CountExpressionComplexity(n.Instance!), //ncrunch: no coverage
			MethodCall m => 1 + (m.Instance != null
				? CountExpressionComplexity(m.Instance)
				: 0) + m.Arguments.Sum(CountExpressionComplexity),
			If i => CountExpressionComplexity(i.Condition) + CountExpressionComplexity(i.Then) +
				(i.OptionalElse != null
					? CountExpressionComplexity(i.OptionalElse)
					: 0),
			_ => 1
		};
}
