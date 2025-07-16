using static Strict.Language.Type;

namespace Strict.Language;

internal class TypeMethodFinder(Type type)
{
	public Type Type { get; } = type;

	public Method? FindFromMethodImplementation(IReadOnlyList<Type> implementationTypes) =>
		!Type.AvailableMethods.TryGetValue(Method.From, out var methods)
			? null
			: methods.FirstOrDefault(m => IsMethodWithMatchingParametersType(m, implementationTypes));

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments)
	{
		var method = FindMethod(methodName, arguments);
		if (method == null)
			throw new NoMatchingMethodFound(Type, methodName, Type.AvailableMethods);
		return method;
	}

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments)
	{
		if (Type.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(Type, Type.Name == Base.Mutable
				? Base.Mutable + " must be used via keyword, not manually constructed!"
				: "Type is Generic and cannot be used directly");
		if (!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return null;
		var typesOfArguments = arguments.Select(argument => argument.ReturnType).ToList();
		var commonTypeOfArguments = TryGetSingleElementType(typesOfArguments);
		foreach (var method in matchingMethods)
			if (IsMethodWithMatchingParametersType(method, typesOfArguments) ||
				commonTypeOfArguments != null &&
				commonTypeOfArguments == GetListElementTypeIfHasSingleParameter(method))
				return method;
		// If this is a from constructor, we can call the methodParameterType constructor to pass
		// along the argument and make it work if it wasn't matching yet.
		if (matchingMethods[0].Parameters.Count == 1 &&
			matchingMethods[0].Parameters[0].Type.FindMethod(Method.From, arguments) != null)
			return matchingMethods[0];
		// Same for enums, no need to create from number, we could just use one of the constants
		// Also allow using numbers for any method that accepts Text as we have Text.From(number)
		if (Type.IsEnum || Type.Name == Base.Text && arguments is [{ ReturnType.Name: Base.Number }])
			return matchingMethods[0];
		throw new ArgumentsDoNotMatchMethodParameters(arguments, Type, matchingMethods);
	}

	private static T? TryGetSingleElementType<T>(IEnumerable<T> argumentTypes) where T : class
	{
		T? firstType = null;
		foreach (var type in argumentTypes)
			if (firstType == null)
				firstType = type;
			else if (firstType != type)
				return null;
		return firstType;
	}

	private static Type? GetListElementTypeIfHasSingleParameter(Method method) =>
		method.Parameters is
		[
			{
				Type: GenericTypeImplementation
				{
					Generic.Name: Base.List
				} parameterType
			}
		]
			? parameterType.ImplementationTypes[0]
			: null;

	private static bool IsMethodWithMatchingParametersType(Method method,
		IReadOnlyList<Type> typesOfArguments)
	{
		//TODO: we can probably just cache the result, no need to go through this every time if the parameters passed in are already correct, which should always be the case anyway!
		if (method is { Name: Method.From, Parameters.Count: 0 } &&
			typesOfArguments.Count == 1 && method.ReturnType.IsSameOrCanBeUsedAs(typesOfArguments[0]))
			return true;
		if (typesOfArguments.Count > method.Parameters.Count || typesOfArguments.Count <
			method.Parameters.Count(p => p.DefaultValue == null))
			return false;
		for (var index = 0; index < typesOfArguments.Count; index++)
			if (!IsMethodParameterMatchingArgument(method, index, typesOfArguments[index]))
				return false;
		return true;
	}

	private static bool IsMethodParameterMatchingArgument(Method method, int index, Type argumentType)
	{
		var methodParameterType = method.Parameters[index].Type;
		if (methodParameterType is GenericTypeImplementation { Generic.Name: Base.Mutable } mutableType)
			methodParameterType = mutableType.ImplementationTypes[0];
		if (argumentType == methodParameterType || method.IsGeneric ||
			methodParameterType.Name == Base.Any ||
			IsArgumentImplementationTypeMatchParameterType(argumentType, methodParameterType))
			return true;
		if (methodParameterType.IsEnum &&
			methodParameterType.Members[0].Type.IsSameOrCanBeUsedAs(argumentType))
			return true;
		if (methodParameterType.Name == Base.Iterator && method.Type.IsSameOrCanBeUsedAs(argumentType))
			return true;
		if (methodParameterType.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(methodParameterType,
				"(parameter " + index + ") is not usable with argument " + argumentType + " in " + method);
		return argumentType.IsSameOrCanBeUsedAs(methodParameterType);
	}

	private static bool
		IsArgumentImplementationTypeMatchParameterType(Type argumentType, Type parameterType) =>
		argumentType is GenericTypeImplementation argumentGenericType &&
		argumentGenericType.ImplementationTypes.Any(t => t == parameterType);
}