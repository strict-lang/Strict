using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Language.Tests")]

namespace Strict.Language;

/// <summary>
/// Methods are parsed lazily, which speeds up type and package parsing enormously and
/// also provides us with all methods in a type usable in any other method if needed.
/// </summary>
public sealed class Method : Context
{
	public Method(Type type, int typeLineNumber, ExpressionParser parser, IReadOnlyList<string> lines)
		: base(type, GetName(lines[0]))
	{
		if (lines.Count > Limit.MethodLength)
			throw new MethodLengthMustNotExceedTwelve(this, lines.Count, typeLineNumber);
		TypeLineNumber = typeLineNumber;
		Parser = parser;
		this.lines = lines;
		var restSpan = lines[0].AsSpan(Name.Length);
		//TODO: measure how much time is wasted at parsing time on error checking and skip it on known good code if it was parsed before and worked (and nothing has changed obviously)
		if (restSpan.StartsWith("()"))
			throw new EmptyParametersMustBeRemoved(this);
		if (restSpan.Length == 1)
			throw new InvalidMethodParameters(this, restSpan.ToString());
		if (IsMethodGeneric(restSpan))
			IsGeneric = true;
		var closingBracketIndex = restSpan.LastIndexOf(") ");
		var returnTypeSpan = closingBracketIndex > 0
			? restSpan[(closingBracketIndex + 2)..]
			: restSpan.Length > 0 && restSpan[0] == ' '
				? restSpan[1..]
				: [];
		ReturnType = returnTypeSpan.Length is 0
			? GetEmptyReturnType(type)
			: ParseReturnType(type, returnTypeSpan.ToString());
		if (lines.Count > 1)
			methodBody = PreParseBody();
		if (restSpan.Length > 2 && restSpan[0] == '(' && closingBracketIndex < 0)
			closingBracketIndex = restSpan.LastIndexOf(")");
		if (closingBracketIndex > 0)
			ParseParameters(type, restSpan[1..closingBracketIndex]);
	}

	public sealed class
		MethodLengthMustNotExceedTwelve(Method method, int linesCount, int lineNumber)
		: ParsingFailed(method.Type, lineNumber,
			$"Method {method.Name} has {linesCount} lines but limit is {Limit.MethodLength}");

	/// <summary>
	/// Simple lexer to just parse the method definition and get all used names and types. Method code
	/// itself is parsed only on demand (when GetBodyAndParseIfNeeded is called) in a more complex
	/// way (Shunting yard/BNF/etc.) and slower. Examples: Run, Run(number), Run returns Text
	/// </summary>
	private static string GetName(ReadOnlySpan<char> firstLine)
	{
		var name = firstLine;
		for (var i = 0; i < firstLine.Length; i++)
			if (firstLine[i] == '(' || firstLine[i] == ' ')
			{
				name = firstLine[..i];
				if (IsNameIsNotOperator(firstLine))
					return firstLine[..(i + 4)].ToString();
				if (IsNameIsNotInOperator(firstLine))
					return firstLine[..(i + 7)].ToString();
				if (IsNameIsInOperator(firstLine))
					return firstLine[..(i + 3)].ToString();
				break;
			}
		return !name.IsWord() && !name.IsOperator()
			? throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name.ToString())
			: name.ToString();
	}

	private static bool IsNameIsInOperator(ReadOnlySpan<char> input) =>
		input.StartsWith(BinaryOperator.IsIn + "(", StringComparison.Ordinal);

	private static bool IsNameIsNotOperator(ReadOnlySpan<char> input) =>
		input.StartsWith(BinaryOperator.IsNot + "(", StringComparison.Ordinal);

	private static bool IsNameIsNotInOperator(ReadOnlySpan<char> input) =>
		input.StartsWith(BinaryOperator.IsNotIn + "(", StringComparison.Ordinal);

	public int TypeLineNumber { get; }
	public ExpressionParser Parser { get; }
	internal readonly IReadOnlyList<string> lines;
	private readonly Body? methodBody;

	private Type ParseReturnType(Context type, string returnTypeText)
	{
		if (returnTypeText == Base.Any)
			throw new MethodReturnTypeAsAnyIsNotAllowed(this, returnTypeText);
		var hasMultipleReturnTypes = returnTypeText.Contains(" or ", StringComparison.Ordinal);
		return hasMultipleReturnTypes
			? ParseMultipleReturnTypes(returnTypeText)
			: type.GetType(returnTypeText);
	}

	private Type ParseMultipleReturnTypes(string typeNames) =>
		new OneOfType(Type, typeNames.Split(" or ", StringSplitOptions.TrimEntries).
			Select(typeName => Type.GetType(typeName)).ToList());

	private Type GetEmptyReturnType(Type type) =>
		Name is From
			? type
			: type.GetType(Base.None);

	public sealed class MethodReturnTypeAsAnyIsNotAllowed(Method method, string name)
		: ParsingFailed(method.Type, 0, name);

	private static bool IsMethodGeneric(ReadOnlySpan<char> headerLine) =>
		headerLine.Contains(Base.Generic, StringComparison.Ordinal) ||
		headerLine.Contains(Base.Generic.MakeFirstLetterLowercase(), StringComparison.Ordinal);

	public bool IsGeneric { get; }

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
			throw new MethodParameterCountMustNotExceedLimit(this,
				TypeLineNumber + methodLineNumber - 1);
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
		nameAndTypeString == Base.Any.MakeFirstLetterLowercase() ||
		nameAndTypeString.Contains(" Any");

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
			: type.GetMemberExpression(parser, nameAndDefaultValue[0], nameAndDefaultValue[1]);
		return //TODO: can't happen: defaultValue == null ? throw new DefaultValueCouldNotBeParsedIntoExpression(this, TypeLineNumber + methodLineNumber - 1, nameAndTypeAsString) :
			new Parameter(type, nameAndDefaultValue[0], defaultValue);
	}

	public sealed class MissingParameterDefaultValue(Method method, int lineNumber,
		string nameAndType) : ParsingFailed(method.Type, lineNumber, nameAndType);

	public sealed class DefaultValueCouldNotBeParsedIntoExpression(Method method,
		int lineNumber,	string defaultValueExpression)
		: ParsingFailed(method.Type, lineNumber, defaultValueExpression);

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

	internal Method(Method cloneFrom, Type newReturnType)
		: base(newReturnType, cloneFrom.Name)
	{
		TypeLineNumber = cloneFrom.TypeLineNumber;
		Parser = cloneFrom.Parser;
		lines = cloneFrom.lines;
		IsGeneric = cloneFrom.IsGeneric;
		ReturnType = newReturnType;
		if (cloneFrom.methodBody != null)
			methodBody = cloneFrom.methodBody.CloneAndUpdateMethod(this);
		parameters = cloneFrom.parameters;
		Tests = cloneFrom.Tests;
		lines = cloneFrom.lines;
	}

	internal Method(Method cloneFrom, GenericTypeImplementation typeWithImplementation)
		: base(typeWithImplementation, cloneFrom.Name)
	{
		TypeLineNumber = cloneFrom.TypeLineNumber;
		Parser = cloneFrom.Parser;
		lines = cloneFrom.lines;
		IsGeneric = false;
		ReturnType = ReplaceWithImplementationOrGenericType(cloneFrom.ReturnType, typeWithImplementation, 0);
		parameters = new List<Parameter>(cloneFrom.parameters);
		for (var index = 0; index < parameters.Count; index++)
			parameters[index] = cloneFrom.parameters[index].CloneWithImplementationType(
				ReplaceWithImplementationOrGenericType(cloneFrom.Parameters[index].Type,
					typeWithImplementation, index));
		if (cloneFrom.methodBody != null)
			methodBody = cloneFrom.methodBody.CloneAndUpdateMethod(this);
		Tests = cloneFrom.Tests;
		lines = cloneFrom.lines;
	}

	private static Type ReplaceWithImplementationOrGenericType(Type type,
		GenericTypeImplementation typeWithImplementation, int index) =>
		type.Name == Base.Generic
			? typeWithImplementation.ImplementationTypes[index] // like Number
			: type.IsGeneric
				? typeWithImplementation // like List(Number)
				: type;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression ParseLine(Body body, string currentLine)
	{
		var expression = Parser.ParseLineExpression(body, currentLine.AsSpan(body.Tabs));
		if (IsTestExpression(currentLine, expression))
			Tests.Add(expression);
		else if (currentLine.Contains(body.Method.Name) &&
			expression.GetType().Name == "MethodCall" &&
			body.ParsingLineNumber == body.Method.Tests.Count + 1 && currentLine != "\tRun")
			throw new RecursiveCallCausesStackOverflow(body);
		return expression;
	}

	private static bool IsTestExpression(string currentLine, Expression expression) =>
		currentLine.Contains($" {BinaryOperator.Is} ") &&
		!currentLine.Trim().StartsWith("if", StringComparison.Ordinal) &&
		!currentLine.Contains("?") && expression.ReturnType.Name == Base.Boolean;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression ParseExpression(Body body, ReadOnlySpan<char> text, bool makeMutable = false) =>
		Parser.ParseExpression(body, text, makeMutable);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> text) =>
		Parser.ParseListArguments(body, text);

	public const string From = "from";

	/// <summary>
	/// Skips the first method declaration line, then counts and removes the tabs from each line.
	/// Also groups all expressions on the same tabs level into bodies. In case a body has only
	/// a single line (which is most often the case), that only expression is used directly.
	/// </summary>
	private Body PreParseBody(int parentTabs = 1, Body? parent = null)
	{
		var body = new Body(this, parentTabs, parent);
		var startLine = methodLineNumber;
		for (; methodLineNumber < lines.Count; methodLineNumber++)
			if (CheckBodyLine(lines[methodLineNumber], body))
				break;
		body.LineRange = new Range(startLine, Math.Min(methodLineNumber, lines.Count));
		return body;
	}

	private int methodLineNumber = 1;

	private bool CheckBodyLine(string line, Body body)
	{
		if (line.Length == 0)
			throw new TypeParser.EmptyLineIsNotAllowed(Type, TypeLineNumber + methodLineNumber);
		var tabs = GetTabs(line);
		if (tabs > body.Tabs)
			PreParseBody(tabs, body);
		CheckIndentation(line, TypeLineNumber + methodLineNumber, tabs);
		return IsCurrentLineInBodyScope(body.Tabs);
	}

	private static int GetTabs(string line)
	{
		var tabs = 0;
		// ReSharper disable once ForCanBeConvertedToForeach, would consume too much memory!
		for (var index = 0; index < line.Length; index++)
			if (line[index] == '\t')
				tabs++;
			else
				break;
		return tabs;
	}

	private void CheckIndentation(string line, int lineNumber, int tabs)
	{
		if (tabs is 0 or > 3)
			throw new InvalidIndentation(Type, lineNumber, line, Name);
		if (char.IsWhiteSpace(line[tabs]))
			throw new TypeParser.ExtraWhitespacesFoundAtBeginningOfLine(Type, lineNumber, line, Name);
		if (char.IsWhiteSpace(line[^1]))
			throw new TypeParser.ExtraWhitespacesFoundAtEndOfLine(Type, lineNumber, line, Name);
	}

	public sealed class InvalidIndentation(Type type, int lineNumber, string line, string method)
		: ParsingFailed(type, lineNumber, method, line);

	private bool IsCurrentLineInBodyScope(int bodyTabs) =>
		methodLineNumber < lines.Count && GetTabs(lines[methodLineNumber]) != bodyTabs;

	public Type Type => (Type)Parent;
	public IReadOnlyList<Parameter> Parameters => parameters;
	private readonly List<Parameter> parameters = new();
	public Type ReturnType { get; }
	public bool IsPublic => char.IsUpper(Name[0]);
	public List<Expression> Tests { get; } = new();

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Base.Value
			? Type
			: Type.FindType(name, searchingFrom ?? this);

	public Expression GetBodyAndParseIfNeeded()
	{
		if (methodBody == null)
			throw new CannotCallBodyOnTraitMethod();
		if (methodBody.Expressions.Count > 0)
			return methodBody; //TODO: currently we only parse once, check if this is ever reachable
		var expression = methodBody.Parse();
		if (Tests.Count < 1 && !IsTestPackage())
			throw new MethodMustHaveAtLeastOneTest(Type, Name, TypeLineNumber);
		return expression;
	}

	private bool IsTestPackage() => Type.Package.Name == "TestPackage" || Name == "Run";

	public sealed class MethodMustHaveAtLeastOneTest(Type type, string name, int typeLineNumber)
		: ParsingFailed(type, typeLineNumber, name);

	public class CannotCallBodyOnTraitMethod : Exception { }

	public override string ToString() =>
		Name + parameters.ToBrackets() + (ReturnType.Name == Base.None
			? ""
			: " " + ReturnType.Name);

	public bool HasEqualSignature(Method method) =>
		Name == method.Name && Parameters.Count == method.Parameters.Count &&
		(ReturnType == method.ReturnType || method.ReturnType.Name == Base.Generic ||
			ReturnType.Name == Base.Generic) && HasSameParameterTypes(method);

	private bool HasSameParameterTypes(Method method) =>
		!method.Parameters.Where((parameter, index) =>
			parameter.Type.Name != Base.Generic && Parameters[index].Type != parameter.Type).Any();

	public int GetParameterUsageCount(string parameterName) =>
		lines.Count(l => l.Contains(" " + parameterName) || l.Contains("(" + parameterName) ||
			l.Contains(parameterName + " ") || l.Contains("\t" + parameterName));

	/// <summary>
	/// Very low level check if a variableName can be found in the raw text of this method lines.
	/// </summary>
	public int GetVariableUsageCount(string variableName) =>
		lines.Count(l => l.Contains(" " + variableName) || l.Contains("(" + variableName) ||
			l.Contains("\t" + variableName));

	/// <summary>
	/// Checks if another method has the same signature, doesn't matter if it is from this type or
	/// any parent or child type. Used to avoid methods with the same return types and parameters.
	/// Slightly different from <see cref="HasEqualSignature"/> which does extra generic checks.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool IsSameMethodNameReturnTypeAndParameters(Method other)
	{
		if (Name != other.Name || ReturnType != other.ReturnType ||
			parameters.Count != other.Parameters.Count)
			return false;
		for (var index = 0; index < parameters.Count; index++)
			if (parameters[index].Type != other.Parameters[index].Type)
				return false;
		return true;
	}
}