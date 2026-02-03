using System.Reflection;
using static Strict.Language.Type;

namespace Strict.Language;

internal class TypeMethodFinder(Type type)
{
	public Type Type { get; } = type;

	public Method? FindFromMethodImplementation(IReadOnlyList<Type> implementationTypes) =>
		!Type.AvailableMethods.TryGetValue(Method.From, out var methods)
			? null
			: methods.FirstOrDefault(m => IsMethodWithMatchingParametersType(m, implementationTypes,
				TryGetSingleElementType(implementationTypes)));

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		FindMethod(methodName, arguments) ??
		throw new NoMatchingMethodFound(Type, methodName, Type.AvailableMethods);

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments)
	{
		if (Type.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(Type, Type.Name == Base.Mutable
				? Base.Mutable + " must be used via keyword, not manually constructed!"
				: "Type is Generic and cannot be used directly");
		return Type is OneOfType
			? FindMethodWithOneOfType(methodName, arguments)
			: FindMethodWithType(methodName, arguments);
	}

	private Method? FindMethodWithOneOfType(string methodName, IReadOnlyList<Expression> arguments)
	{
		ArgumentsDoNotMatchMethodParameters? lastArgumentsMismatchException = null;
		foreach (var subType in ((OneOfType)Type).Types)
		{
			try
			{
				var foundSubTypeMethod = subType.FindMethod(methodName, arguments);
				if (foundSubTypeMethod != null)
					return foundSubTypeMethod;
			}
			catch (ArgumentsDoNotMatchMethodParameters argumentsMismatch)
			{
				lastArgumentsMismatchException = argumentsMismatch;
			}
		}
		return lastArgumentsMismatchException != null
			? throw lastArgumentsMismatchException
			: null;
	}

	private Method? FindMethodWithType(string methodName, IReadOnlyList<Expression> arguments)
	{
		if (!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return null;
		if (arguments.Count == 1 && arguments[0].ReturnType.Name == Base.Error)
			return matchingMethods[0];
		var typesOfArguments = arguments.Select(argument => argument.ReturnType).ToList();
		var commonTypeOfArguments = TryGetSingleElementType(typesOfArguments);
		foreach (var method in matchingMethods)
			if (IsMethodWithMatchingParametersType(method, typesOfArguments, commonTypeOfArguments) ||
				commonTypeOfArguments != null && commonTypeOfArguments ==
				GetListElementTypeIfHasSingleParameter(method, arguments.Count))
				return method;
		// Single character text can always be used as a character (thus number)
		if (arguments.Count == 1 && matchingMethods.Count > 0 &&
			matchingMethods[0].Parameters.Count > 0 &&
			matchingMethods[0].Parameters[0].Type.Name is Base.Number or Base.Character &&
			arguments[0].ReturnType.Name == Base.Text && arguments[0].IsConstant &&
			arguments[0].GetType().Name == "Text" && GetTextValue(arguments[0]).Length == 1)
			return matchingMethods[0];
		// If this is a from constructor, we can call the methodParameterType constructor to pass
		// along the argument and make it work if it wasn't matching yet.
		if (methodName == Method.From && matchingMethods[0].Parameters.Count == 1)
		{
			var innerFromMethod =
				matchingMethods[0].Parameters[0].Type.FindMethod(Method.From, arguments);
			if (innerFromMethod != null &&
				IsFromConstructorWithMatchingConstraints(matchingMethods[0], arguments.Count))
				return matchingMethods[0];
		}
		if (IsAnyIsComparison(methodName, arguments, matchingMethods))
			return matchingMethods[0];
		throw new ArgumentsDoNotMatchMethodParameters(arguments, Type, matchingMethods);
	}

	/// <summary>
	/// Allow `is`/`is not` comparisons against Any even if the argument is not literally Any.
	/// This is used by data types like Range in TestPackage where equality is defined as `is(other Any)`.
	/// </summary>
	private static bool IsAnyIsComparison(string methodName, IReadOnlyList<Expression> arguments,
		IReadOnlyList<Method> matchingMethods) =>
		methodName is BinaryOperator.Is or UnaryOperator.Not &&
		arguments.Count == 1 &&
		matchingMethods.Count > 0 &&
		matchingMethods[0].Parameters is [{ Type.Name: Base.Any }];

	private static string GetTextValue(Expression argument) =>
		argument.GetType().GetProperty("Data", BindingFlags.Instance | BindingFlags.Public)?.
			GetValue(argument)?.ToString() ?? "";

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

	private static Type? GetListElementTypeIfHasSingleParameter(Method method,
		int numberOfArguments) =>
		method.Parameters is
		[
			{
				Type: GenericTypeImplementation
				{
					Generic.Name: Base.List
				} parameterType
			}
		] && IsFromConstructorWithMatchingConstraints(method, numberOfArguments)
			? parameterType.ImplementationTypes[0]
			: null;

	private static bool IsFromConstructorWithMatchingConstraints(Method method, int numberOfArguments)
	{
		if (method.Name != Method.From)
			return true;
		var member = method.Type.Members.FirstOrDefault(m => !m.IsConstant && m.Type.Name != Base.Iterator);
		return member?.Constraints == null ||
			member.Constraints[0].ToString().Contains("Length is " + numberOfArguments); //TODO: do actual evaluation of constraint
	}

	private static bool IsMethodWithMatchingParametersType(Method method,
		IReadOnlyList<Type> typesOfArguments, Type? commonTypeOfArguments)
	{
		// Don't check trait methods, but allow Run for tests and from constructors
		if (method.IsTrait && method.Name != Base.Run && method.Name != Method.From &&
			// Also is type checks are ok and some types are not implemented, but done at the runtime!
			commonTypeOfArguments?.Name != Base.Type && method.Parent.Name != Base.File &&
			// Casting to types must be allowed as well (is always a trait, never implemented)
			(method.Name != BinaryOperator.To || method.ReturnType.Name != Base.Type) &&
			// Enum number values can always be compared
			(method.Name != BinaryOperator.Is || commonTypeOfArguments?.Name != Base.Number))
			return false;
		if (method is { Name: Method.From, Parameters.Count: 0 } && typesOfArguments.Count == 1 &&
			method.ReturnType.IsSameOrCanBeUsedAs(typesOfArguments[0], false))
			return true; //ncrunch: no coverage
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
			IsArgumentImplementationTypeMatchParameterType(argumentType, methodParameterType))
			return true;
		if (methodParameterType.Name != Base.Text && methodParameterType.IsEnum &&
			methodParameterType.Members[0].Type.IsSameOrCanBeUsedAs(argumentType))
			return true;
		if (methodParameterType.Name == Base.Iterator && method.Type.IsSameOrCanBeUsedAs(argumentType))
			return true; //ncrunch: no coverage
		if (methodParameterType.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(methodParameterType, //ncrunch: no coverage
				"(parameter " + index + ") is not usable with argument " + argumentType + " in " + method);
		return argumentType.IsSameOrCanBeUsedAs(methodParameterType);
	}

	private static bool
		IsArgumentImplementationTypeMatchParameterType(Type argumentType, Type parameterType) =>
		argumentType is GenericTypeImplementation argumentGenericType &&
		argumentGenericType.ImplementationTypes.Any(t => t == parameterType);
}