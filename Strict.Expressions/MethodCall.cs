using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Any type of method we can call, this includes normal local method calls, recursions, calls to
/// any of our implement base types (instance is null in all of those cases), calls to other types
/// (either From(type) or instance method calls, there are no static methods) or any operator
/// <see cref="Binary"/> or <see cref="Not"/> unary call (which are all normal methods as well).
/// Like MemberCall has the same syntax when the parent instance is used: Type.Method
/// </summary>
public class MethodCall : ConcreteExpression
{
	public MethodCall(Method method, Expression? instance = null, IReadOnlyList<Expression>? arguments = null,
		Type? toReturnType = null, int lineNumber = 0) :
		base(GetMethodReturnType(method, toReturnType), lineNumber, method.ReturnType.IsMutable)
	{
		if (method.Name == Method.From && instance != null)
			throw new CannotCallFromConstructorWithExistingInstance(); //ncrunch: no coverage
		Instance = instance;
		Method = method;
		Arguments = arguments ?? [];
	}

	public sealed class CannotCallFromConstructorWithExistingInstance : Exception;

	private static Type GetMethodReturnType(Method method, Type? toReturnType) =>
		method.Name == BinaryOperator.To && toReturnType != null
			? toReturnType
			: method.ReturnType;

	public Method Method { get; }
	public Expression? Instance { get; }
	public IReadOnlyList<Expression> Arguments { get; }
	public override bool IsConstant =>
		(Instance?.IsConstant ?? true) && Arguments.All(a => a.IsConstant);

	// ReSharper disable once TooManyArguments
	public static Expression? TryParse(Expression? instance, Body body,
		IReadOnlyList<Expression> arguments, Type type, string inputAsString)
	{
		if (body.IsFakeBodyForMemberInitialization)
			return null;
		var method = type.FindMethod(inputAsString, arguments) ??
			(instance == null && type == body.Method.Type
				? FindPrivateMethod(type, inputAsString, arguments)
				: null);
		if (method == null)
			return null;
		return new MethodCall(method, instance, AreArgumentsAutoParsedAsList(method, arguments)
			? [new List(body, (List<Expression>)arguments)]
			: arguments, null, body.CurrentFileLineNumber);
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
			if (parameterType is GenericTypeImplementation { Generic.Name: Base.Mutable } mutableType)
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
		var fromType = body.Method.FindType(methodName);
		return fromType is null
			? null
			: IsConstructorUsedWithSameArgumentType(arguments, fromType)
				? body.IsFakeBodyForMemberInitialization && arguments.Count == 1
					? arguments[0]
					: throw new ConstructorForSameTypeArgumentIsNotAllowed(body)
				: CreateFromMethodCall(body, fromType, arguments);
	}

	internal static Expression CreateFromMethodCall(Body body, Type fromType,
		IReadOnlyList<Expression> arguments, Expression? basedOnErrorVariable = null)
	{
		fromType = NormalizeListAndDictionaryImplementation(fromType, arguments);
		ValidateMutableImplementation(fromType, arguments);
		arguments = NormalizeDictionaryArguments(body, fromType, arguments);
		arguments = NormalizeErrorArguments(body, ref fromType, arguments, basedOnErrorVariable);
		arguments = NormalizeTypeArguments(body, fromType, arguments);
		return new MethodCall(fromType.GetMethod(Method.From, arguments), null, arguments, null,
			body.CurrentFileLineNumber);
	}

	private static Type NormalizeListAndDictionaryImplementation(Type fromType,
		IReadOnlyList<Expression> arguments)
	{
		if (fromType.IsList && fromType.IsGeneric && arguments.Count > 0)
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
		if (fromType.IsMutable && fromType.IsGeneric && args.Count == 1 &&
			args[0].ReturnType is not GenericTypeImplementation { Generic.Name: Base.List })
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(fromType,
				Base.Mutable + " must be used with a List implementation");
		if (fromType is GenericTypeImplementation { Generic.Name: Base.Mutable } mutableImpl &&
			mutableImpl.ImplementationTypes[0] is not GenericTypeImplementation
			{
				Generic.Name: Base.List
			})
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(mutableImpl,
				Base.Mutable + " must be used with a List implementation");
	}

	private static IReadOnlyList<Expression> NormalizeDictionaryArguments(Body body, Type fromType,
		IReadOnlyList<Expression> arguments)
	{
		if (!fromType.IsDictionary)
			return arguments;
		if (arguments.Count > 1)
			return [new List(body, arguments.ToList())];
		return arguments.Count == 1 && arguments[0] is List { Values.Count: 2 } singlePair
			? [new List(body, [singlePair])]
			: arguments;
	}

	private static IReadOnlyList<Expression> NormalizeErrorArguments(Body body, ref Type fromType,
		IReadOnlyList<Expression> arguments, Expression? basedOnErrorVariable)
	{
		if (!fromType.IsSameOrCanBeUsedAs(fromType.GetType(Base.Error)))
			return arguments;
		if (arguments.Count == 0)
			return
			[
				new Value(body.Method.GetType(Base.Name),
					basedOnErrorVariable?.ToString() ?? (fromType.IsError
						? body.CurrentDeclarationNameForErrorText ?? body.Method.Name
						: fromType.Name)),
				CreateListFromMethodCall(body, Base.Stacktrace, CreateStacktraces(body))
			];
		if (arguments.Count > 1)
			throw new Type.ArgumentsDoNotMatchMethodParameters(arguments, fromType, fromType.Methods); //ncrunch: no coverage
		if (basedOnErrorVariable != null)
		{
			fromType = fromType.GetType(Base.ErrorWithValue).
				GetGenericImplementation(arguments[0].ReturnType);
			return [basedOnErrorVariable, arguments[0]];
		}
		if (arguments[0] is Value { ReturnType.Name: Base.Text } textValue)
			return
			[
				new Value(body.Method.GetType(Base.Name), textValue.Data.ToString() ?? ""),
				CreateListFromMethodCall(body, Base.Stacktrace, CreateStacktraces(body))
			];
		arguments =
		[
			CreateFromMethodCall(body, fromType, []),
			arguments[0]
		];
		fromType = fromType.GetType(Base.ErrorWithValue).
			GetGenericImplementation(arguments[1].ReturnType);
		return arguments;
	}

	private static IReadOnlyList<Expression> NormalizeTypeArguments(Body body, Type fromType,
		IReadOnlyList<Expression> arguments) =>
		fromType.Name == Base.Type && arguments.Count == 1
			? [
				arguments[0].ReturnType.IsText
					? new Value(body.Method.GetType(Base.Name), ((Value)arguments[0]).Data)
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
		CreateFromMethodCall(body, body.Method.GetType(Base.Stacktrace), [
			CreateFromMethodCall(body, body.Method.GetType(Base.Method), [
				new Value(body.Method.GetType(Base.Name), body.Method.Name),
				CreateFromMethodCall(body, body.Method.GetType(Base.Type),
					[new Text(body.Method, body.Method.Type.Name)])
			]),
			new Text(body.Method, body.Method.Type.FilePath),
			new Number(body.Method, body.ParsingLineNumber)
		]);

	private static bool
		IsConstructorUsedWithSameArgumentType(IReadOnlyList<Expression> arguments, Type fromType) =>
		arguments.Count is 1 && (fromType == arguments[0].ReturnType ||
			arguments[0].ReturnType is GenericTypeImplementation genericType && fromType == genericType.Generic);

	public sealed class ConstructorForSameTypeArgumentIsNotAllowed(Body body) : ParsingFailed(body);

	public override string ToString() =>
		Instance is not null
			? (Instance is Binary
				? $"({Instance})"
				: $"{Instance}") + $".{Method.Name}{Arguments.ToBrackets()}"
			: ReturnType is GenericTypeImplementation { Generic.Name: Base.ErrorWithValue }
				? Arguments[0] + "(" + Arguments[1] + ")"
				: ReturnType.IsError
					? Base.Error
					: Method.Name == Method.From &&
					ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary }
						? FormatDictionaryConstructor()
						: $"{GetProperMethodName()}{Arguments.ToBrackets()}";

	private string FormatDictionaryConstructor() =>
		Arguments.Count == 1 && Arguments[0] is List list
			? Base.Dictionary + (list.Values.All(value => value is List)
				? list.Values.ToBrackets()
				: $"({list})")
			: throw new NotSupportedException("Invalid Dictionary arguments: " + Arguments.ToBrackets());

	private string GetProperMethodName() =>
		Method.Name == Method.From
			? Method.ReturnType.Name
			: Method.Name;
}