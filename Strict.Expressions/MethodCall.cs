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
		if (method != null)
			return new MethodCall(method, instance, AreArgumentsAutoParsedAsList(method, arguments)
				? [new List(body, (List<Expression>)arguments)]
				: arguments, null, body.CurrentFileLineNumber);
		return null;
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
		if (fromType.Name == Base.List && fromType.IsGeneric && arguments.Count > 0)
			fromType = fromType.GetGenericImplementation(arguments[0].ReturnType);
    if (fromType.Name == Base.Mutable && fromType.IsGeneric && arguments.Count == 1 &&
			arguments[0].ReturnType is not GenericTypeImplementation { Generic.Name: Base.List })
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(fromType,
				Base.Mutable + " must be used with a List implementation");
    if (fromType is GenericTypeImplementation { Generic.Name: Base.Mutable } mutableImpl &&
			mutableImpl.ImplementationTypes[0] is not GenericTypeImplementation
			{
				Generic.Name: Base.List
			})
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(mutableImpl,
				Base.Mutable + " must be used with a List implementation");
   if (fromType.Name == Base.Dictionary && fromType.IsGeneric)
		{
      if (arguments.Count > 1)
				fromType = arguments[0] is List { Values.Count: 2 } firstPair
					? fromType.GetGenericImplementation(firstPair.Values[0].ReturnType,
						firstPair.Values[1].ReturnType)
					: fromType.GetGenericImplementation(arguments[0].ReturnType,
						arguments[1].ReturnType);
			else if (arguments.Count > 0 && arguments[0] is List { Values.Count: 2 } pair)
				fromType = fromType.GetGenericImplementation(pair.Values[0].ReturnType,
					pair.Values[1].ReturnType);
		}
		if ((fromType.Name == Base.Dictionary ||
			fromType is GenericTypeImplementation { Generic.Name: Base.Dictionary }) &&
      arguments.Count > 1)
			arguments = [new List(body, arguments.ToList())];
		else if ((fromType.Name == Base.Dictionary ||
			fromType is GenericTypeImplementation { Generic.Name: Base.Dictionary }) &&
			arguments.Count == 1 && arguments[0] is List { Values.Count: 2 } singlePair)
			arguments = [new List(body, [singlePair])];
		// For Error always fill in Name and Stacktraces and use ErrorWithValue if argument is given
		if (fromType.IsSameOrCanBeUsedAs(fromType.GetType(Base.Error)))
		{
			if (arguments.Count == 0)
				arguments =
				[
					new Value(body.Method.GetType(Base.Name), basedOnErrorVariable?.ToString() ??
						(fromType.Name == Base.Error
							? body.CurrentDeclarationNameForErrorText ?? body.Method.Name
							: fromType.Name)),
					CreateListFromMethodCall(body, Base.Stacktrace, CreateStacktraces(body))
				];
			else if (arguments.Count > 1)
				throw new Type.ArgumentsDoNotMatchMethodParameters(arguments, fromType, fromType.Methods);
			else if (basedOnErrorVariable != null)
			{
				arguments =
				[
					basedOnErrorVariable,
					arguments[0]
				];
				fromType = fromType.GetType(Base.ErrorWithValue).
					GetGenericImplementation(arguments[1].ReturnType);
			}
			else if (arguments[0] is Value { ReturnType.Name: Base.Text } textValue)
			{
				arguments =
				[
					new Value(body.Method.GetType(Base.Name), textValue.Data.ToString() ?? ""),
					CreateListFromMethodCall(body, Base.Stacktrace, CreateStacktraces(body))
				];
			}
			else
			{
				arguments =
				[
					CreateFromMethodCall(body, fromType, []),
					arguments[0]
				];
				fromType = fromType.GetType(Base.ErrorWithValue).
					GetGenericImplementation(arguments[1].ReturnType);
			}
		}
		// Type can fill in Package automatically (but you need to give the name)
		else if (fromType.Name == Base.Type && arguments.Count == 1)
			arguments =
			[
				arguments[0].ReturnType.Name == Base.Text
					? new Value(body.Method.GetType(Base.Name), ((Value)arguments[0]).Data)
					: arguments[0],
				new Text(body.Method, body.Method.Type.Package.FullName)
			];
		return new MethodCall(fromType.GetMethod(Method.From, arguments), null, arguments, null,
			body.CurrentFileLineNumber);
	}

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
				: ReturnType.Name == Base.Error
					? Base.Error
         : Method.Name == Method.From &&
						ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary }
							? FormatDictionaryConstructor()
							: $"{GetProperMethodName()}{Arguments.ToBrackets()}";

	private string FormatDictionaryConstructor()
	{
		if (Arguments.Count == 1 && Arguments[0] is List list)
		{
			var argumentText = list.Values.All(value => value is List)
				? list.Values.ToBrackets()
				: $"({list})";
			return Base.Dictionary + argumentText;
		}
		return Base.Dictionary + Arguments.ToBrackets();
	}

	private string GetProperMethodName() =>
		Method.Name == Method.From
			? Method.ReturnType.Name
			: Method.Name;
}