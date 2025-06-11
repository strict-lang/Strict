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
		ReturnType = restSpan.Length is 0
			? GetEmptyReturnType(type)
			: ParseReturnType(type, restSpan);
		if (lines.Count > 1)
			methodBody = PreParseBody();
		if (restSpan.Length is not 0)
			ParseParameters(type, restSpan);
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

	private Type ParseReturnType(Type type, ReadOnlySpan<char> rest)
	{
		if (IsReturnTypeAny(rest))
			throw new MethodReturnTypeAsAnyIsNotAllowed(this, rest.ToString());
		if (IsMethodGeneric(rest))
			IsGeneric = true;
		var closingBracketIndex = rest.LastIndexOf(')');
		while (closingBracketIndex > 1 && rest[closingBracketIndex - 1] == ')')
			closingBracketIndex--;
		/*I am very confused about this, if there is a ) bracket before the last opening one, we search for the last space and cut that off and use that as our return type? what?
		 e.g. (newLine = Character(13)) would return Character(13)), which crashes at the 13) number
		var lastOpeningBracketIndex = rest.LastIndexOf('(');
		if (lastOpeningBracketIndex > 2 && rest.IndexOf(')') < lastOpeningBracketIndex)
			return Type.GetType(rest[(rest.LastIndexOf(' ') + 1)..].ToString());
		 */
		var hasMultipleReturnTypes = rest.Contains(" or ", StringComparison.Ordinal);
		return closingBracketIndex > 0 && rest.Length == 2
			? throw new EmptyParametersMustBeRemoved(this)
			: rest.Length < 2
				? throw new InvalidMethodParameters(this, rest.ToString())
				: rest[0] is ' '
					? hasMultipleReturnTypes
						? ParseMultipleReturnTypes(rest[1..].ToString())
						: type.GetType(rest[1..].ToString())
					: closingBracketIndex + 2 < rest.Length
						? hasMultipleReturnTypes
							? ParseMultipleReturnTypes(rest[(closingBracketIndex + 2)..].ToString())
							: Type.GetType(rest[(closingBracketIndex + 2)..].ToString())
						: GetEmptyReturnType(type);
	}

	private Type ParseMultipleReturnTypes(string typeNames) =>
		new OneOfType(Type, typeNames.Split(" or ", StringSplitOptions.TrimEntries).
			Select(typeName => Type.GetType(typeName)).ToList());

	private Type GetEmptyReturnType(Type type) =>
		Name is From
			? type
			: type.GetType(Base.None);

	private static bool IsReturnTypeAny(ReadOnlySpan<char> rest) =>
		rest[0] is ' ' && rest[1..].Equals(Base.Any, StringComparison.Ordinal);

	public sealed class MethodReturnTypeAsAnyIsNotAllowed(Method method, string name)
		: ParsingFailed(method.Type, 0, name);

	private static bool IsMethodGeneric(ReadOnlySpan<char> headerLine) =>
		headerLine.Contains(Base.Generic, StringComparison.Ordinal) ||
		headerLine.Contains(Base.Generic.MakeFirstLetterLowercase(), StringComparison.Ordinal);

	public bool IsGeneric { get; private set; }

	private void ParseParameters(Type type, ReadOnlySpan<char> rest)
	{
		var closingBracketIndex = rest.LastIndexOf(')');
		var lastOpeningBracketIndex = rest.LastIndexOf('(');
		// If the type contains brackets, exclude it from the rest for proper parameter parsing
		if (lastOpeningBracketIndex > 2)
		{
			var lastSpaceIndex = rest.LastIndexOf(' ');
			if (lastSpaceIndex > 0)
				ParseAndAddParameters(type, rest, rest[closingBracketIndex - 1] == ')'
					? closingBracketIndex
					: lastSpaceIndex - 1);
			return;
		}
		if (closingBracketIndex > 0)
			ParseAndAddParameters(type, rest, closingBracketIndex);
	}

	private void ParseAndAddParameters(Type type, ReadOnlySpan<char> rest, int closingBracketIndex)
	{
		foreach (var nameAndType in SplitParameters(rest, closingBracketIndex))
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

	private static SpanSplitEnumerator SplitParameters(ReadOnlySpan<char> rest, int closingBracketIndex)
	{
		var parametersSpan = rest[1..closingBracketIndex];
		return parametersSpan.Contains('(') && (!parametersSpan.Contains(',') || IsCommaInsideBrackets(parametersSpan, parametersSpan.IndexOf(',')))
			? new SpanSplitEnumerator(parametersSpan, char.MaxValue, StringSplitOptions.None)
			: parametersSpan.Split(',', StringSplitOptions.TrimEntries);
	}

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
		return defaultValue == null
			? throw new DefaultValueCouldNotBeParsedIntoExpression(this,
				TypeLineNumber + methodLineNumber - 1, nameAndTypeAsString)
			: new Parameter(type, nameAndDefaultValue[0], defaultValue);
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
	public Expression ParseExpression(Body body, ReadOnlySpan<char> text) =>
		Parser.ParseExpression(body, text);

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
		//tst: Console.WriteLine("PreParseBody " + this);
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
		foreach (var t in line)
			if (t == '\t')
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
	private List<Parameter> parameters = new();
	public Type ReturnType { get; private set; }
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
		{
			Console.WriteLine(methodBody + " existing expressions=" +
				methodBody.Expressions.ToWordList());
			return methodBody;
		}
		if (methodBody.Method != this)
			throw new NotSupportedException("methodBody is not matching this method anymore " + this);
		Console.WriteLine(methodBody + " parse");
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

	public Method CloneFrom(Type concreteType)
	{
		var clone = (Method)MemberwiseClone();
		clone.Parent = concreteType;
		clone.ReturnType = concreteType;
		clone.methodBody?.UpdateCurrentAndChildrenMethod(clone);
		return clone;
	}

	public Method CloneWithImplementation(GenericTypeImplementation typeWithImplementation)
	{
		var clone = (Method)MemberwiseClone();
		clone.ReturnType = ReplaceWithImplementationOrGenericType(clone.ReturnType, typeWithImplementation, 0);
		clone.parameters = new List<Parameter>(parameters);
		for (var index = 0; index < clone.Parameters.Count; index++)
			clone.parameters[index] = clone.parameters[index].CloneWithImplementationType(
				ReplaceWithImplementationOrGenericType(clone.Parameters[index].Type,
					typeWithImplementation, index));
		clone.Parent = typeWithImplementation; //TODO: find any alternative way to have method with updated parent?
		clone.IsGeneric = false;
		clone.methodBody?.UpdateCurrentAndChildrenMethod(clone);
		return clone;
	}

	private static Type ReplaceWithImplementationOrGenericType(Type type,
		GenericTypeImplementation typeWithImplementation, int index) =>
		type.Name == Base.Generic
			? typeWithImplementation.ImplementationTypes[index] // like Number
			: type.IsGeneric
				? typeWithImplementation // like List(Number)
				: type;

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

	public int GetVariableUsageCount(string variableName) =>
		lines.Count(l => l.Contains(" " + variableName) || l.Contains("(" + variableName) ||
			l.Contains("\t" + variableName));

	/// <summary>
	/// Checks if another method has the same signature, doesn't matter if it is from this type or
	/// any parent or child type. Used to avoid methods with the same return types and parameters.
	/// Slightly different from <see cref="HasEqualSignature"/> which does extra generic checks.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool IsSameMethodNameReturnTypeAndParameters(Method other) =>
		Name == other.Name && ReturnType == other.ReturnType &&
		Parameters.SequenceEqual(other.Parameters);
}