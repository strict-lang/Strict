using static Strict.Language.Type;

namespace Strict.Language;

internal class TypeMethodFinder(Type type)
{
	public Type Type { get; } = type;

	public Method? FindMethod(string methodName, IReadOnlyList<Type> implementationTypes) =>
		!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods)
			? null
			: matchingMethods.FirstOrDefault(method =>
				IsMethodWithMatchingParametersType(method, implementationTypes));

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
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(Type,
				"Type is Generic and cannot be used directly");
		if (!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return null;
		var typesOfArguments = arguments.Select(argument => argument.ReturnType).ToList();
		var commonTypeOfArguments = TryGetSingleElementType(typesOfArguments);
		foreach (var method in matchingMethods)
			if (IsMethodWithMatchingParametersType(method, typesOfArguments) ||
				commonTypeOfArguments != null &&
				commonTypeOfArguments == GetListElementTypeIfHasSingleParameter(method))
				return method;
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
		IReadOnlyList<Type> argumentReturnTypes)
	{
		if (method is { Name: Method.From, Parameters.Count: 0 } &&
			argumentReturnTypes.Count == 1 && method.ReturnType.IsCompatible(argumentReturnTypes[0]))
			return true;
		if (method.Parameters.Count != argumentReturnTypes.Count)
			return false;
		for (var index = 0; index < method.Parameters.Count; index++)
			if (!IsMethodParameterMatchingArgument(method, index, argumentReturnTypes[index]))
				return false;
		return true;
	}

	private static bool IsMethodParameterMatchingArgument(Method method, int index,
		Type argumentReturnType)
	{
		var methodParameterType = method.Parameters[index].Type;
		if (argumentReturnType == methodParameterType || method.IsGeneric ||
			methodParameterType.Name == Base.Any ||
			IsArgumentImplementationTypeMatchParameterType(argumentReturnType, methodParameterType))
			return true;
		if (methodParameterType.IsEnum &&
			methodParameterType.Members[0].Type.IsCompatible(argumentReturnType))
			return true;
		if (methodParameterType.Name == Base.Iterator && method.Type.IsCompatible(argumentReturnType))
			return true;
		if (methodParameterType.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(
				methodParameterType, //ncrunch: no coverage
				"(parameter " + index + ") is not usable with argument " + argumentReturnType + " in " +
				method);
		return argumentReturnType.IsCompatible(methodParameterType);
	}

	private static bool
		IsArgumentImplementationTypeMatchParameterType(Type argumentType, Type parameterType) =>
		argumentType is GenericTypeImplementation argumentGenericType &&
		argumentGenericType.ImplementationTypes.Any(t => t == parameterType);
}