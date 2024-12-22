using static Strict.Language.Type;

namespace Strict.Language;

internal class TypeMethodFinder
{
	public TypeMethodFinder(Type type) => Type = type;
	public Type Type { get; }

	public Method? FindMethod(string methodName, IReadOnlyList<Type> implementationTypes) =>
		!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods)
			? null
			: matchingMethods.FirstOrDefault(method =>
				method.Parameters.Count == implementationTypes.Count &&
				IsMethodWithMatchingParametersType(method, implementationTypes));

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments, ExpressionParser parser) =>
		FindMethod(methodName, arguments, parser) ??
		throw new NoMatchingMethodFound(Type, methodName, Type.AvailableMethods);

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments,
		ExpressionParser parser)
	{
		if (Type.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(Type,
				"Type is Generic and cannot be used directly");
		if (!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return FindAndCreateFromBaseMethod(methodName, arguments, parser);
		foreach (var method in matchingMethods)
		{
			if (method.Parameters.Count == arguments.Count)
			{
				//TODO: clean up, optimized this a bit
				if (arguments.Count == 1)
				{
					if (IsMethodParameterMatchingArgument(method, 0, arguments[0].ReturnType))
						return method;
				}
				else if (IsMethodWithMatchingParametersType(method,
					arguments.Select(argument => argument.ReturnType).ToList()))
					return method;
			}
			//TODO: not sure about this, looks very slow to do on every possible method
			if (method.Parameters.Count == 1 && arguments.Count > 0)
			{
				var parameter = method.Parameters[0];
				if (IsParameterTypeList(parameter) && CanAutoParseArgumentsIntoList(arguments) &&
					IsMethodParameterMatchingWithArgument(arguments,
						(GenericTypeImplementation)parameter.Type))
					return method;
			}
		}
		return FindAndCreateFromBaseMethod(methodName, arguments, parser) ??
			throw new ArgumentsDoNotMatchMethodParameters(arguments, Type, matchingMethods);
	}

	private static bool IsParameterTypeList(NamedType parameter) =>
		parameter.Type is GenericTypeImplementation { Generic.Name: Base.List };

	private static bool CanAutoParseArgumentsIntoList(IReadOnlyList<Expression> arguments) =>
		arguments.All(a => a.ReturnType == arguments[0].ReturnType);

	private static bool IsMethodParameterMatchingWithArgument(IReadOnlyList<Expression> arguments,
		GenericTypeImplementation genericType) =>
		genericType.ImplementationTypes[0] == arguments[0].ReturnType;

	//TODO: got two usages, but they are different and can be optimized each
	private static bool IsMethodWithMatchingParametersType(Method method,
		IReadOnlyList<Type> argumentReturnTypes)
	{
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
		if (methodParameterType.IsEnum && methodParameterType.Members[0].Type == argumentReturnType)
			return true;
		if (methodParameterType.Name == Base.Iterator && method.Type == argumentReturnType)
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

	private Method? FindAndCreateFromBaseMethod(string methodName,
		IReadOnlyList<Expression> arguments, ExpressionParser parser)
	{
		if (methodName != Method.From)
			return null;
		var fromMethod = "from(";
		fromMethod += GetMatchingMemberParametersIfExist(arguments);
		return fromMethod.Length > 5 && (fromMethod.Split(',').Length - 1 == arguments.Count ||
			fromMethod.Split(',').Length - 1 == PrivateMembersCount)
			? BuildMethod($"{fromMethod[..^2]})", parser)
			: Type.IsDataType
				? BuildMethod(fromMethod[..^1], parser)
				: null;
	}

	private string? GetMatchingMemberParametersIfExist(IReadOnlyList<Expression> arguments)
	{
		var argumentIndex = 0;
		string? parameters = null;
		foreach (var member in Type.Members)
			if (arguments.Count > argumentIndex && member.Type == arguments[argumentIndex].ReturnType)
			{
				parameters += $"{member.Name.MakeFirstLetterLowercase()} {member.Type.Name}, ";
				argumentIndex++;
			}
		return parameters == null && arguments.Count > 1 && PrivateMembersCount == 1 &&
			CanUpcastAllArgumentsToMemberType(arguments, Type.Members[0], arguments[0].ReturnType)
				? FormatMemberAsParameterWithType()
				: parameters;
	}

	private int PrivateMembersCount => Type.Members.Count(member => !member.IsPublic);

	private static bool CanUpcastAllArgumentsToMemberType(IEnumerable<Expression> arguments,
		NamedType member, Type firstArgumentReturnType) =>
		member.Type is GenericTypeImplementation { Generic.Name: Base.List } genericType &&
		genericType.ImplementationTypes[0] == firstArgumentReturnType &&
		arguments.All(a => a.ReturnType == firstArgumentReturnType);

	private string FormatMemberAsParameterWithType() =>
		$"{Type.Members[0].Name.MakeFirstLetterLowercase()} {Type.Members[0].Type.Name}, ";

	private Method BuildMethod(string fromMethod, ExpressionParser parser) =>
		new(Type, 0, parser, new[] { fromMethod });
}