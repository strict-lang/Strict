using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Language.Tests")]
[assembly: InternalsVisibleTo("Strict.Validators")]
[assembly: InternalsVisibleTo("Strict.HighLevelRuntime")]
[assembly: InternalsVisibleTo("Strict.Bytecode")]

namespace Strict.Language;

public sealed partial class Method
{
	private void ParseParameters(Type type, ReadOnlySpan<char> parametersSpan)
	{
		foreach (var nameAndType in SplitParameters(parametersSpan))
		{
			if (char.IsUpper(nameAndType[0]))
				throw new ParametersMustStartWithLowerCase(this, nameAndType.ToString());
			var nameAndTypeAsString = nameAndType.ToString();
			if (IsParameterTypeAny(nameAndTypeAsString))
				throw new ParametersWithTypeAnyIsNotAllowed(this, nameAndTypeAsString);
			parameters.Add(nameAndTypeAsString.Contains('=')
				? GetParameterByExtractingNameAndDefaultValue(type, nameAndTypeAsString, Parser)
				: new Parameter(type, nameAndTypeAsString));
		}
		if (parameters.Count > Limit.ParameterCount)
			throw new MethodParameterCountMustNotExceedLimit(this, TypeLineNumber + methodLineNumber - 1);
	}

	private static SpanSplitEnumerator SplitParameters(ReadOnlySpan<char> parametersSpan) =>
		parametersSpan.Contains('(') && (!parametersSpan.Contains(',') ||
			IsCommaInsideBrackets(parametersSpan, parametersSpan.IndexOf(',')))
			? new SpanSplitEnumerator(parametersSpan, char.MaxValue, StringSplitOptions.None)
			: parametersSpan.Split(',', StringSplitOptions.TrimEntries);

	private static bool IsCommaInsideBrackets(ReadOnlySpan<char> parametersSpan, int commaIndex) =>
		parametersSpan.IndexOf(')') > commaIndex && parametersSpan.LastIndexOf('(') < commaIndex;

	public sealed class ParametersMustStartWithLowerCase(Method method, string message)
		: ParsingFailed(method.Type, 0, message, method.Name);

	private static bool IsParameterTypeAny(string nameAndTypeString) =>
		nameAndTypeString == Type.AnyLowercase || nameAndTypeString.Contains(" " + Type.Any);

	public sealed class ParametersWithTypeAnyIsNotAllowed(Method method, string name)
		: ParsingFailed(method.Type, 0, name);

	private Parameter GetParameterByExtractingNameAndDefaultValue(Type type,
		string nameAndTypeAsString, ExpressionParser parser)
	{
		var nameAndDefaultValue = nameAndTypeAsString.Split(" = ");
		if (nameAndDefaultValue.Length < 2)
			throw new MissingParameterDefaultValue(this, TypeLineNumber + methodLineNumber - 1,
				nameAndTypeAsString);
		var defaultValue = methodBody != null
			? ParseExpression(methodBody, nameAndDefaultValue[1])
			: type.GetMemberExpression(parser, nameAndDefaultValue[0], nameAndDefaultValue[1],
				TypeLineNumber);
		return new Parameter(type, nameAndDefaultValue[0], defaultValue);
	}

	public sealed class MissingParameterDefaultValue(Method method,
		int lineNumber,
		string nameAndType) : ParsingFailed(method.Type, lineNumber, nameAndType);

	public sealed class MethodParameterCountMustNotExceedLimit(Method method, int lineNumber)
		: ParsingFailed(method.Type, lineNumber,
			$"{
				GetMethodName(method)
			} has parameters count {
				method.Parameters.Count
			} but limit is {
				Limit.ParameterCount
			}")
	{
		private static string GetMethodName(Method method) =>
			method.Name == From
				? "Type " + method.Type.FullName + " " + From + " constructor method"
				: "Method " + method.Name;
	}

	public sealed class InvalidMethodParameters(Method method, string rest)
		: ParsingFailed(method.Type, 0, rest, method.Name);

	public sealed class EmptyParametersMustBeRemoved(Method method)
		: ParsingFailed(method.Type, 0, "", method.Name);
}
