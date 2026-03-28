using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Any type of method we can call, this includes normal local method calls, recursions, calls to
/// any of our implement base types (instance is null in all of those cases). Calls to other types
/// (either From(type) or instance method calls, there are no static methods) or any operator
/// <see cref="Binary"/> or <see cref="Not"/> unary call (which are all normal methods as well).
/// Like MemberCall has the same syntax when the parent instance is used: Type.Method
/// </summary>
public class MethodCall : ConcreteExpression
{
	public MethodCall(Method method, Expression? instance = null, IReadOnlyList<Expression>? arguments = null,
		Type? toReturnType = null, int lineNumber = 0) :
    base(GetMethodReturnType(method, toReturnType, instance), lineNumber, method.ReturnType.IsMutable)
	{
		if (method.Name == Method.From && instance != null)
			throw new CannotCallFromConstructorWithExistingInstance(); //ncrunch: no coverage
		Instance = instance;
		Method = method;
		Arguments = arguments ?? [];
	}

	public sealed class CannotCallFromConstructorWithExistingInstance : Exception;

 private static Type GetMethodReturnType(Method method, Type? toReturnType, Expression? instance)
	{
		if (method.Name == BinaryOperator.To && toReturnType != null)
			return toReturnType;
		var returnType = method.ReturnType;
    if (instance?.ReturnType is not Type instanceType || !IsConcreteListShape(instanceType))
			return returnType;
		if (returnType.IsList && returnType.IsGeneric)
			return instanceType;
    if (returnType is GenericTypeImplementation
			{
				Generic.Name: Type.Mutable,
				ImplementationTypes: [var mutableInnerType]
			} mutableListReturn && IsGenericListShape(mutableInnerType))
			return mutableListReturn.Generic.GetGenericImplementation(instanceType);
		return returnType;
	}

	private static bool IsConcreteListShape(Type type) =>
		(type.IsList || type is GenericType { Generic.Name: Type.List } ||
			type is GenericTypeImplementation { Generic.Name: Type.List }) && !type.IsGeneric;

	private static bool IsGenericListShape(Type type) =>
		type is GenericType { Generic.Name: Type.List } ||
		type is GenericTypeImplementation { Generic.Name: Type.List, IsGeneric: true } ||
		type.IsList && type.IsGeneric;

	public Method Method { get; }
	public Expression? Instance { get; }
	public IReadOnlyList<Expression> Arguments { get; }
	public override bool IsConstant
	{
		get
		{
			if (Instance?.IsConstant == false)
				return false;
			for (var index = 0; index < Arguments.Count; index++)
				if (!Arguments[index].IsConstant)
					return false;
			return true;
		}
	}

	protected string AddNestedBracketsIfNeeded(Expression child, int addPrecedenceForNot = 0) =>
		child is MethodCall binaryOrUnary && BinaryOperator.GetPrecedence(binaryOrUnary.Method.Name) <
		BinaryOperator.GetPrecedence(Method.Name) + addPrecedenceForNot || child is If
			? $"({child})"
			: child.ToString();

	// ReSharper disable once TooManyArguments
	public static Expression? TryParse(Expression? instance, Body body,
		IReadOnlyList<Expression> arguments, Type type, string inputAsString)
	{
		if (body.IsFakeBodyForMemberInitialization)
			return null;
		Method? method;
		try
		{
			method = type.FindMethod(inputAsString, arguments) ??
				(type == body.Method.Type
					? FindPrivateMethod(type, inputAsString, arguments)
					: null);
		}
		catch (Type.GenericTypesCannotBeUsedDirectlyUseImplementation exception)
		{
			method = TryFindMethodOnCurrentGenericType(body, type, instance, inputAsString, arguments);
			if (method == null)
				throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(exception,
					GetMethodLookupContext(body, inputAsString, instance, arguments));
		}
		if (method == null)
			return null;
		return new MethodCall(method, instance, AreArgumentsAutoParsedAsList(method, arguments)
			? [new List(body, (List<Expression>)arguments)]
			: arguments, null, body.CurrentFileLineNumber);
	}

	private static Method? TryFindMethodOnCurrentGenericType(Body body, Type type,
		Expression? instance, string inputAsString, IReadOnlyList<Expression> arguments) =>
		!type.IsGeneric || type is GenericTypeImplementation ||
		type != body.Method.Type && instance?.ReturnType != type
			? null
			: FindPrivateMethod(type, inputAsString, arguments);

	private static string GetMethodLookupContext(Body body, string inputAsString,
		Expression? instance, IReadOnlyList<Expression> arguments)
	{
		var context = body.Method.Type + "." + body.Method.Name + " at line " +
			(body.CurrentFileLineNumber + 1) + ", source: " + body.CurrentLine.Trim() +
			", lookup instance: " + (instance?.ToString() ?? "none") +
			", lookup method: " + inputAsString +
			", lookup arguments: " + (arguments.Count == 0
				? "none"
				: string.Join(", ", arguments.Select(argument =>
					argument + " => " + argument.ReturnType)));
		var line = body.CurrentLine.Trim();
		var plusIndex = line.IndexOf(" + ", StringComparison.Ordinal);
		var isIndex = line.IndexOf(" is ", StringComparison.Ordinal);
		if (plusIndex <= 0 || isIndex <= plusIndex + 3)
			return context;
		try
		{
			var plusInstance = body.Method.ParseExpression(body, line.AsSpan(0, plusIndex));
			var plusArgument = body.Method.ParseExpression(body,
				line.AsSpan(plusIndex + 3, isIndex - plusIndex - 3));
			return context + ", + instance type: " + plusInstance.ReturnType +
				", + argument[0] type: " + plusArgument.ReturnType;
		}
		catch (ParsingFailed)
		{
			return context;
		}
	}

	private static Method? FindPrivateMethod(Type type, string methodName,
		IReadOnlyList<Expression> arguments)
	{
		foreach (var candidate in type.Methods)
			if (candidate.Name == methodName && AreArgumentsCompatible(candidate, arguments))
				return candidate;
		return null;
	}

	private static bool AreArgumentsCompatible(Method method, IReadOnlyList<Expression> arguments)
	{
		if (arguments.Count > method.Parameters.Count || arguments.Count <
			method.Parameters.Count(p => p.DefaultValue == null))
			return false;
		for (var index = 0; index < arguments.Count; index++)
		{
			var parameterType = method.Parameters[index].Type;
			if (parameterType is GenericTypeImplementation { Generic.Name: Type.Mutable } mutableType)
				parameterType = mutableType.ImplementationTypes[0];
			if (!arguments[index].ReturnType.IsSameOrCanBeUsedAs(parameterType))
				return false;
		}
		return true;
	}

	private static bool
		AreArgumentsAutoParsedAsList(Method method, IReadOnlyCollection<Expression> arguments) =>
		method.Parameters.Count != arguments.Count && method.Parameters.Count is 1 &&
		arguments.Count > 1;

	public static Expression? TryParseFromOrEnum(Body body, IReadOnlyList<Expression> arguments,
		string methodName)
	{
		if (!methodName.AsSpan().IsWordOrWordWithNumberAtEnd(out _))
			return null;
		Type? fromType;
		try
		{
			fromType = body.Method.FindType(methodName);
		}
		catch (Context.TypeNotFound)
		{
			return null;
		}
		if (fromType == null)
			return null;
		if (fromType.IsList && arguments.Count == 0)
			return new List(fromType, body.CurrentFileLineNumber);
		return IsConstructorUsedWithSameArgumentType(arguments, fromType)
			? body.IsFakeBodyForMemberInitialization && arguments.Count == 1
				? arguments[0]
				: throw new ConstructorForSameTypeArgumentIsNotAllowed(body, arguments, fromType)
			: CreateFromMethodCall(body, fromType, arguments);
	}

	internal static Expression CreateFromMethodCall(Body body, Type fromType,
		IReadOnlyList<Expression> arguments, Expression? basedOnErrorVariable = null)
	{
		fromType = NormalizeListAndDictionaryImplementation(fromType, arguments);
		(fromType, arguments) = NormalizeMutableImplementationAndArguments(body, fromType, arguments);
		ValidateMutableImplementation(fromType, arguments);
		arguments = NormalizeDictionaryArguments(body, fromType, arguments);
		arguments = NormalizeErrorArguments(body, ref fromType, arguments, basedOnErrorVariable);
		arguments = NormalizeTypeArguments(body, fromType, arguments);
		var method = fromType.GetMethod(Method.From, arguments);
		if (AreArgumentsAutoParsedAsList(method, arguments))
			arguments = [new List(body, (List<Expression>)arguments)];
		return new MethodCall(method, null, arguments, null, body.CurrentFileLineNumber);
	}

	private static (Type fromType, IReadOnlyList<Expression> arguments)
		NormalizeMutableImplementationAndArguments(Body body, Type fromType,
			IReadOnlyList<Expression> arguments)
	{
		if (!fromType.IsMutable || arguments.Count != 1)
			return (fromType, arguments);
		if (fromType.IsGeneric && fromType is not GenericTypeImplementation)
			fromType = fromType.GetGenericImplementation(arguments[0].ReturnType);
		return (fromType, arguments);
	}

	private static Type NormalizeListAndDictionaryImplementation(Type fromType,
		IReadOnlyList<Expression> arguments)
	{
		if (fromType is { IsList: true, IsGeneric: true } && arguments.Count > 0)
			return fromType.GetGenericImplementation(arguments[0].ReturnType);
		if (!fromType.IsDictionary || !fromType.IsGeneric)
			return fromType;
		if (arguments.Count > 1)
			return arguments[0] is List { Values.Count: 2 } firstPair
				? fromType.GetGenericImplementation(firstPair.Values[0].ReturnType,
					firstPair.Values[1].ReturnType)
				: fromType.GetGenericImplementation(arguments[0].ReturnType, arguments[1].ReturnType);
		if (arguments.Count > 0 && arguments[0] is List { Values.Count: 2 } pair)
			return fromType.GetGenericImplementation(pair.Values[0].ReturnType,
				pair.Values[1].ReturnType);
		return fromType; //ncrunch: no coverage
	}

	private static void ValidateMutableImplementation(Type fromType, IReadOnlyList<Expression> args)
	{
		if (fromType is { IsMutable: true, IsGeneric: true } && fromType is not GenericTypeImplementation)
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(fromType,
				Type.Mutable + " must use a concrete implementation type");
	}

	private static IReadOnlyList<Expression> NormalizeDictionaryArguments(Body body, Type fromType,
		IReadOnlyList<Expression> arguments)
	{
		if (!fromType.IsDictionary)
			return arguments;
		if (arguments.Count > 1)
			return [new List(body, arguments.ToList())];
		return arguments is [List { Values.Count: 2 } singlePair]
			? [new List(body, [singlePair])]
			: arguments;
	}

	private static IReadOnlyList<Expression> NormalizeErrorArguments(Body body, ref Type fromType,
		IReadOnlyList<Expression> arguments, Expression? basedOnErrorVariable)
	{
		if (!fromType.IsSameOrCanBeUsedAs(fromType.GetType(Type.Error)))
			return arguments;
		if (arguments.Count == 0)
			return
			[
				new Value(body.Method.GetType(nameof(Type.Name)),
					basedOnErrorVariable?.ToString() ?? (fromType.IsError
						? body.CurrentDeclarationNameForErrorText ?? body.Method.Name
						: fromType.Name)),
				CreateListFromMethodCall(body, Type.Stacktrace, CreateStacktraces(body))
			];
		if (arguments.Count > 1)
			throw new Type.ArgumentsDoNotMatchMethodParameters(arguments, fromType, fromType.Methods); //ncrunch: no coverage
		if (basedOnErrorVariable != null)
		{
			fromType = fromType.GetType(Type.ErrorWithValue).
				GetGenericImplementation(arguments[0].ReturnType);
			return [basedOnErrorVariable, arguments[0]];
		}
		if (arguments[0] is Value { ReturnType.Name: Type.Text } textValue)
			return
			[
				new Value(body.Method.GetType(nameof(Type.Name)), textValue.Data.Text),
				CreateListFromMethodCall(body, Type.Stacktrace, CreateStacktraces(body))
			];
		arguments = [CreateFromMethodCall(body, fromType, []), arguments[0]];
		fromType = fromType.GetType(Type.ErrorWithValue).
			GetGenericImplementation(arguments[1].ReturnType);
		return arguments;
	}

	private static IReadOnlyList<Expression> NormalizeTypeArguments(Body body, Type fromType,
		IReadOnlyList<Expression> arguments) =>
		fromType.Name == nameof(Type) && arguments.Count == 1
			? [
				arguments[0].ReturnType.IsText
					? new Value(body.Method.GetType(nameof(Type.Name)), ((Value)arguments[0]).Data)
					: arguments[0],
				new Text(body.Method, body.Method.Type.Package.FullName)
			]
			: arguments;

	private static Expression CreateListFromMethodCall(Body body, string listElementTypeName,
		IReadOnlyList<Expression> arguments) =>
		CreateFromMethodCall(body,
			body.Method.GetListImplementationType(body.Method.GetType(listElementTypeName)), arguments);

	private static IReadOnlyList<Expression> CreateStacktraces(Body body) =>
		[CreateStacktrace(body)];

	private static Expression CreateStacktrace(Body body) =>
		CreateFromMethodCall(body, body.Method.GetType(Type.Stacktrace), [
			CreateFromMethodCall(body, body.Method.GetType(nameof(Method)), [
				new Value(body.Method.GetType(nameof(Type.Name)), body.Method.Name),
				CreateFromMethodCall(body, body.Method.GetType(nameof(Type)),
					[new Text(body.Method, body.Method.Type.Name)])
			]),
			new Text(body.Method, body.Method.Type.FilePath),
			new Number(body.Method, body.ParsingLineNumber)
		]);

	private static bool
		IsConstructorUsedWithSameArgumentType(IReadOnlyList<Expression> arguments, Type fromType) =>
		arguments.Count is 1 && (fromType == arguments[0].ReturnType ||
			arguments[0].ReturnType is GenericTypeImplementation genericType &&
			fromType == genericType.Generic);

	public sealed class ConstructorForSameTypeArgumentIsNotAllowed(Body body,
		IReadOnlyList<Expression> arguments, Type fromType) : ParsingFailed(body,
			"Don't construct this type " + fromType + " with itself, arguments: " + arguments.ToBrackets());

	public override string ToString() =>
		Instance is not null && Instance.ToString() != Type.ValueLowercase
			? (Instance is Binary
				? $"({Instance})"
				: $"{Instance}") + $".{Method.Name}{Arguments.ToBrackets()}"
			: ReturnType is GenericTypeImplementation { Generic.Name: Type.ErrorWithValue }
				? Arguments[0] + "(" + Arguments[1] + ")"
				: ReturnType.IsError
					? FormatErrorConstructor()
					: Method.Name == Method.From &&
					ReturnType is GenericTypeImplementation { Generic.Name: Type.Dictionary }
						? FormatDictionaryConstructor()
						: Method.Name == Method.From && IsAutoWrappedListArgument()
							? $"{GetProperMethodNameWithFromSupport()}({string.Join(", ", ((List)Arguments[0]).Values)})"
							: $"{GetProperMethodNameWithFromSupport()}{Arguments.ToBrackets()}";

	private string FormatErrorConstructor()
	{
		if (Arguments.Count != 2 || !Arguments[0].ReturnType.IsText)
			return Type.Error;
		var errorText = Arguments[0].ToString();
		return errorText.Length > 1 && errorText[0] == '"' && errorText[^1] == '"' &&
			!errorText[1..^1].IsWord()
				? $"{Type.Error}({Arguments[0]})"
				: Type.Error;
	}

	private bool IsAutoWrappedListArgument() =>
		Arguments is [List { Values.Count: > 1 }] &&
		Method.Parameters.Count == 1 &&
		Method.Parameters[0].Type.IsList &&
		!ReturnType.IsList;

	private string GetProperMethodNameWithFromSupport() =>
		Method.Name == Method.From
			? ReturnType is GenericTypeImplementation { Generic.Name: Type.Mutable }
				? Type.Mutable
				: Method.ReturnType.Name
			: Method.Name;

	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) ||
		other is MethodCall mc && other.GetType() == GetType() &&
		Method.IsSameMethodNameReturnTypeAndParameters(mc.Method) &&
		Equals(Instance, mc.Instance) && ArgumentsEqual(mc.Arguments);

	private bool ArgumentsEqual(IReadOnlyList<Expression> otherArguments)
	{
		if (Arguments.Count != otherArguments.Count)
			return false; //ncrunch: no coverage
		for (var index = 0; index < Arguments.Count; index++)
			if (!Arguments[index].Equals(otherArguments[index]))
				return false; //ncrunch: no coverage
		return true;
	}

	public override int GetHashCode() => Method.GetHashCode() ^ (Instance?.GetHashCode() ?? 0);

	private string FormatDictionaryConstructor() =>
		Arguments is [List list]
			? Type.Dictionary + (list.Values.All(value => value is List)
				? list.Values.ToBrackets()
				: $"({list})")
     : throw new InvalidDictionaryArgumentsForFormatting(Method, Arguments);

	private sealed class InvalidDictionaryArgumentsForFormatting(Method method,
		IReadOnlyList<Expression> arguments)
		: ParsingFailed(method.Type, method.TypeLineNumber,
			"Invalid Dictionary arguments: " + arguments.ToBrackets(), method.ToString());
}