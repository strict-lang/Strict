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

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments,
		ExpressionParser parser) =>
		FindMethod(methodName, arguments, parser) ??
		throw new NoMatchingMethodFound(Type, methodName, Type.AvailableMethods);

	//TODO: method too long
	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments,
		ExpressionParser parser)
	{
		if (Type.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(Type,
				"Type is Generic and cannot be used directly");
		if (!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return FindAndCreateFromBaseMethod(methodName, arguments, parser);
		var typesOfArguments =
			new Lazy<IReadOnlyList<Type>>(() =>
				arguments.Select(argument => argument.ReturnType).ToList());
		//TODO: explain what this does, put it into a method name, we don't want to use multiple different types of arguments here
		//old code had problems, this fixed it, but can be simplified: var commonTypeOfArguments = new Lazy<Type?>(() => TrySingle(typesOfArguments.Value.Distinct()));
		var commonTypeOfArguments =		
			new Lazy<Type?>(() =>
			{
				Type? result = null;
				foreach (var type in typesOfArguments.Value.Distinct())
				{
					if (result != null)
						return null;
					result = type;
				}
				return result;
			});
		foreach (var method in matchingMethods)
		{
			if (IsMethodWithMatchingParametersType(method, typesOfArguments.Value))
				return method;
			if (commonTypeOfArguments.Value != null && commonTypeOfArguments.Value ==
				GetListElementTypeIfHasSingleParameter(method))
				return method;
		}
		return FindAndCreateFromBaseMethod(methodName, arguments, parser) ??
			throw new ArgumentsDoNotMatchMethodParameters(arguments, Type, matchingMethods);
	}

	/// <summary>
	/// if <see cref="list"/> contains exactly one item it returns this item
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="list"></param>
	/// <returns>if <see cref="list"/> contains exactly one item it returns this item, otherwise it returns the default value</returns>
	private static T? TrySingle<T>(IEnumerable<T> list)
	{
		T? result = default;
		foreach (var item in list)
		{
			if (!ReferenceEquals(result, default))
				return default;
			result = item;
		}
		return result;
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
		var matchingMemberParameters = GetMatchingMemberParametersIfExist(arguments);
		if (string.IsNullOrEmpty(matchingMemberParameters))
			return null;
		var fromMethod = "from(" + matchingMemberParameters;
		var length = matchingMemberParameters.Split(',').Length - 1;
		return length == arguments.Count || length == PrivateMembersCount
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
		new(Type, 0, parser, [fromMethod]);
}