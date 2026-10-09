using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Language.Tests")]
[assembly: InternalsVisibleTo("Strict.Validators")]
[assembly: InternalsVisibleTo("Strict.HighLevelRuntime")]
[assembly: InternalsVisibleTo("Strict.Bytecode")]

namespace Strict.Language;

/// <summary>
/// Methods are parsed lazily, which speeds up type and package parsing enormously and
/// also provides us with all methods in a type usable in any other method if needed.
/// </summary>
public sealed partial class Method : Context
{
#if DEBUG
	public Method(Type type, int typeLineNumber, ExpressionParser parser, IReadOnlyList<string> lines,
		[CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = "") : base(type, GetName(lines[0]), callerFilePath,
		callerLineNumber, callerMemberName)

#else
	public Method(Type type, int typeLineNumber, ExpressionParser parser, IReadOnlyList<string> lines)
		: base(type, GetName(lines[0]))
#endif
	{
		if (lines.Count > Limit.MethodLength)
			throw new MethodLengthMustNotExceedTwelve(this, lines.Count, typeLineNumber);
		TypeLineNumber = typeLineNumber;
		Parser = parser;
		this.lines = lines;
		var restSpan = lines[0].AsSpan(Name.Length);
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

	public sealed class MethodLengthMustNotExceedTwelve(Method method, int linesCount, int lineNumber)
		: ParsingFailed(method.Type, lineNumber,
			$"Method {method.Name} has {linesCount} lines but limit is {Limit.MethodLength}");

	/// <summary>
	/// Simple lexer to just parse the method definition and get all used names and types. Method
	/// code itself is parsed only on demand (when GetBodyAndParseIfNeeded is called) in a more
	/// complex way (Shunting yard/BNF/etc.) and slower. Examples: Run, Run(number), Run returns Text
	/// </summary>
	private static string GetName(ReadOnlySpan<char> firstLine)
	{
		var name = firstLine;
		for (var i = 0; i < firstLine.Length; i++)
			if (firstLine[i] == '(' || firstLine[i] == ' ')
			{
				name = firstLine[..i];
				break;
			}
		return !name.IsWord() && !name.IsOperator()
			? throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name.ToString())
			: name.ToString();
	}

	public int TypeLineNumber { get; }

	public ExpressionParser Parser { get; }

	internal readonly IReadOnlyList<string> lines;

	private readonly Body? methodBody;

	public bool WasParsedAlready => methodBody is { Expressions.Count: > 0 };

	private Type ParseReturnType(Context type, string returnTypeText)
	{
		if (returnTypeText == Type.Any)
			throw new MethodReturnTypeAsAnyIsNotAllowed(this, returnTypeText);
		var hasMultipleReturnTypes = returnTypeText.Contains(" or ", StringComparison.Ordinal);
		return hasMultipleReturnTypes
			? ParseMultipleReturnTypes(type, returnTypeText)
			: type.GetType(returnTypeText);
	}

	private Type ParseMultipleReturnTypes(Context type, string typeNames)
	{
		var splitNames = typeNames.Split(" or ", StringSplitOptions.TrimEntries);
		var types = new Type[splitNames.Length];
		for (var index = 0; index < types.Length; index++)
			types[index] = Type.GetType(splitNames[index]);
		var typeName = OneOfType.BuildName(types);
		return type.FindType(typeName) ?? new OneOfType(Type, types, typeName);
	}

	private Type GetEmptyReturnType(Type type) =>
		Name is From
			? type
			: type.GetType(Type.None);

	public sealed class MethodReturnTypeAsAnyIsNotAllowed(Method method, string name)
		: ParsingFailed(method.Type, 0, name);

	private static bool IsMethodGeneric(ReadOnlySpan<char> headerLine) =>
		headerLine.Contains(Type.GenericUppercase, StringComparison.Ordinal) ||
		headerLine.Contains(Type.GenericLowercase, StringComparison.Ordinal);

	public bool IsGeneric { get; }

	internal Method(Method cloneFrom, Type newReturnType) : base(newReturnType, cloneFrom.Name
#if DEBUG
		, cloneFrom.callerFilePath, cloneFrom.callerLineNumber, cloneFrom.callerMemberName
#endif
	)
	{
		TypeLineNumber = cloneFrom.TypeLineNumber;
		Parser = cloneFrom.Parser;
		lines = cloneFrom.lines;
		IsGeneric = cloneFrom.IsGeneric;
		ReturnType = newReturnType;
		if (cloneFrom.methodBody != null)
			methodBody = cloneFrom.methodBody.CloneAndUpdateMethod(this); //ncrunch: no coverage
		parameters = cloneFrom.parameters;
		Tests = cloneFrom.Tests;
		lines = cloneFrom.lines;
	}

	internal Method(Method cloneFrom, GenericTypeImplementation typeWithImplementation) : base(
		typeWithImplementation, cloneFrom.Name
#if DEBUG
		, cloneFrom.callerFilePath, cloneFrom.callerLineNumber, cloneFrom.callerMemberName
#endif
	)
	{
		TypeLineNumber = cloneFrom.TypeLineNumber;
		Parser = cloneFrom.Parser;
		lines = cloneFrom.lines;
		IsGeneric = false;
		ReturnType =
			ReplaceWithImplementationOrGenericType(cloneFrom.ReturnType, typeWithImplementation, 0);
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

	/// <summary>
	/// Nested generic implementations (like Mutable(List)) keep their outer type while substituting
	/// the implemented list type. This makes Add(Type) return Mutable(List(Number)) as expected.
	/// </summary>
	private static Type ReplaceWithImplementationOrGenericType(Type type,
		GenericTypeImplementation typeWithImplementation, int index)
	{
		if (type.Name == Type.GenericUppercase)
			return typeWithImplementation.ImplementationTypes[index];
		if (type is GenericTypeImplementation genericImplementation)
		{
			var updatedImplementationTypes = new Type[genericImplementation.ImplementationTypes.Count];
			var hasChanges = false;
			for (var implementationIndex = 0; implementationIndex < updatedImplementationTypes.Length;
				implementationIndex++)
			{
				var implementationType = genericImplementation.ImplementationTypes[implementationIndex];
				var updatedType = implementationType.Name == Type.GenericUppercase
					? typeWithImplementation.ImplementationTypes[index]
					: implementationType == typeWithImplementation.Generic
						? typeWithImplementation
						: implementationType;
				updatedImplementationTypes[implementationIndex] = updatedType;
				if (!ReferenceEquals(updatedType, implementationType))
					hasChanges = true;
			}
			if (hasChanges)
				return genericImplementation.Generic.GetGenericImplementation(updatedImplementationTypes);
		}
		return type.IsGeneric && type == typeWithImplementation.Generic
			? typeWithImplementation
			: type;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression ParseLine(Body body, string currentLine)
	{
		var expression = Parser.ParseLineExpression(body, currentLine.AsSpan(body.Tabs));
		if (IsTestExpression(body, currentLine, expression))
			Tests.Add(expression);
		// Checks for obvious recursive calls with same arguments at the last line (non-test method),
		// this won't catch most recursive calls, see Executor for most other cases.
		else if (expression.GetType().Name == "MethodCall" &&
			body.ParsingLineNumber == body.Method.Tests.Count + 1 && currentLine != "\tRun" &&
			(currentLine == body.Method.GetNameWithParameters() ||
				currentLine.EndsWith("." + body.Method.GetNameWithParameters(), StringComparison.Ordinal) ||
				currentLine.Contains("." + body.Method.GetNameWithParameters() + " ") ||
				currentLine.Contains(body.Method.GetNameWithParameters() + ".")) &&
			!IsCallOnOtherTypedInstance(body, currentLine))
			throw new RecursiveCallCausesStackOverflow(body);
		return expression;
	}

	private bool IsCallOnOtherTypedInstance(Body body, string currentLine)
	{
		var line = currentLine.AsSpan().TrimStart();
		var nameEnd = line.IndexOfAny('.', '(');
		if (nameEnd <= 0)
			return false;
		var instanceName = line[..nameEnd].ToString();
		var instanceType = Type.FindMember(instanceName)?.Type ?? body.FindVariable(instanceName)?.Type ??
			parameters.FirstOrDefault(parameter => parameter.Name == instanceName)?.Type ??
			(char.IsUpper(instanceName[0])
				? Type.FindType(instanceName)
				: null);
		return instanceType != null && instanceType != Type;
	}

	private string GetNameWithParameters()
	{
		var result = "";
		foreach (var param in parameters)
			result += (result == ""
				? ""
				: ", ") + param.Type.Name;
		if (result.Length > 0)
			result = "(" + result + ")";
		return Name + result;
	}

	public sealed class RecursiveCallCausesStackOverflow(Body body) : ParsingFailed(body);

	private static bool IsTestExpression(Body body, string currentLine, Expression expression) =>
		!IsLastMethodLine(body) &&
		(currentLine.Contains($" {BinaryOperator.Is} ") || (expression.GetType().Name == "MethodCall" &&
			body.ParsingLineNumber == body.Method.Tests.Count + 1 &&
			(body.Parent != null || body.ParsingLineNumber < body.LineRange.End.Value - 1))) &&
		!currentLine.Trim().StartsWith("if ", StringComparison.Ordinal) &&
		!currentLine.Contains(" then ") && expression.ReturnType.IsBoolean;

	private static bool IsLastMethodLine(Body body) =>
		body.Parent == null && body.ParsingLineNumber == body.LineRange.End.Value - 1;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression ParseExpression(Body body, ReadOnlySpan<char> text, bool makeMutable = false) =>
		Parser.ParseExpression(body, text, makeMutable);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> text) =>
		Parser.ParseListArguments(body, text);

	public const string From = "from";

	public const string Run = nameof(Run);

	private int methodLineNumber = 1;

	private readonly object parseLock = new();

	public Type Type => (Type)Parent;

	public IReadOnlyList<Parameter> Parameters => parameters;

	private readonly List<Parameter> parameters = new();

	public Type ReturnType { get; }

	public bool IsPublic => char.IsUpper(Name[0]);

	public List<Expression> Tests { get; } = new();

	public bool IsTrait => methodBody == null;

	public override Type? FindTypeCore(string name, Context? searchingFrom = null) =>
		name == Type.ValueLowercase
			? Type
			: Type.FindTypeCore(name, searchingFrom ?? this);

	public Expression GetBodyAndParseIfNeeded(bool parseTestsOnlyForGeneric = false)
	{
		if (methodBody == null)
			throw new CannotCallBodyOnTraitMethod(Type, Name);
		lock (parseLock)
		{
			if (parseTestsOnlyForGeneric && Type.IsGeneric)
				return ParseTestsOnlyForGeneric();
			if (methodBody.Expressions.Count > 0)
				return !parseTestsOnlyForGeneric && methodBody.Expressions.Any(expression =>
					expression.GetType().Name == nameof(PlaceholderExpression))
					? methodBody.Parse()
					: methodBody.Expressions.Count == 1
						? methodBody.Expressions[0]
						: methodBody;
			var expression = methodBody.Parse();
			if (expression.GetType().Name == Body.Declaration)
				throw new DeclarationIsNeverUsedAndMustBeRemoved(Type, TypeLineNumber, expression);
			if (methodBody.Variables != null)
				foreach (var variable in methodBody.Variables)
					if (variable is { IsMutable: true, InitialValue.IsConstant: true } &&
						!Parser.IsVariableMutated(methodBody, variable.Name))
						throw new MutableUsesConstantValue(methodBody, variable.Name, variable.InitialValue);
			return BodyParsed?.Invoke(expression) ?? expression;
		}
	}

	public sealed class DeclarationIsNeverUsedAndMustBeRemoved(Type type,
		int lineNumber,
		Expression expression) : ParsingFailed(type, lineNumber, expression.ToString());

	public sealed class MutableUsesConstantValue(Body body, string name, Expression value)
		: ParsingFailed(body,
			$"Mutable declaration uses constant value, use constant instead: constant {name} = {value}");

	/// <summary>
	/// Needed when rewriting method body to a single expression, or creating a new Body from Visitor.
	/// </summary>
	internal void SetBodySingleExpression(Expression expression) =>
		methodBody!.SetExpressions(expression is Body body
			? body.Expressions
			: [expression]);

	public event Func<Expression, Expression>? BodyParsed;

	public class CannotCallBodyOnTraitMethod(Type type, string name) : Exception(
		type.Name + "." + name + " is a trait method and has no implementation");

	public override string ToString() =>
		Name + parameters.ToBrackets() + (ReturnType.IsNone
			? ""
			: " " + ReturnType.Name);

	public bool HasEqualSignature(Method method) =>
		Name == method.Name && Parameters.Count == method.Parameters.Count &&
		(ReturnType == method.ReturnType || method.ReturnType.Name == Type.GenericUppercase ||
			ReturnType.Name == Type.GenericUppercase) && HasSameParameterTypes(method);

	private bool HasSameParameterTypes(Method method) =>
		!method.Parameters.Where((parameter, index) => parameter.Type.Name != Type.GenericUppercase &&
			Parameters[index].Type != parameter.Type).Any();

	public int GetParameterUsageCount(string parameterName) =>
		lines.Count(l => l.Contains(" " + parameterName) || l.Contains("(" + parameterName) ||
			l.Contains(parameterName + " ") || l.Contains("\t" + parameterName));

	/// <summary>
	/// Very low level check if a variableName can be found in the raw text in these method lines.
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
