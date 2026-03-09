using System.Reflection;
using static Strict.Language.Type;

namespace Strict.Language;

internal class TypeMethodFinder(Type type)
{
	public Type Type { get; } = type;

	public Method? FindFromMethodImplementation(IReadOnlyList<Type> implementationTypes) =>
		Type.AvailableMethods.TryGetValue(Method.From, out var methods)
			? FindFromMethod(implementationTypes, methods)
			: null;

	private Method? FindFromMethod(IReadOnlyList<Type> implementationTypes, List<Method> methods)
	{
		for (var index = 0; index < methods.Count; index++)
			if (IsMethodWithMatchingParametersType(methods[index], implementationTypes,
				TryGetSingleElementType(implementationTypes), Type))
				return methods[index];
		return null; //ncrunch: no coverage
	}

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		FindMethod(methodName, arguments) ??
		throw new NoMatchingMethodFound(Type, methodName, Type.AvailableMethods);

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments)
	{
		if (Type.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(Type, Type.IsMutable
				? Mutable + " must be used via keyword, not manually constructed!"
				: "Type is Generic and cannot be used directly");
		if (Type is OneOfType oneOfType)
		{
			foreach (var subType in oneOfType.Types)
			{
				var foundSubTypeMethod = subType.FindMethod(methodName, arguments);
				if (foundSubTypeMethod != null)
					return foundSubTypeMethod;
			} //ncrunch: no coverage
			return null; //ncrunch: no coverage
		}
		return FindMethodWithType(methodName, arguments);
	}

	private Method? FindMethodWithType(string methodName, IReadOnlyList<Expression> arguments)
	{
		if (!Type.AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return null;
		if (arguments is [{ ReturnType.IsError: true }, _])
			return matchingMethods[0];
		var typesOfArguments = arguments.Select(argument => argument.ReturnType).ToList();
		var commonTypeOfArguments = TryGetSingleElementType(typesOfArguments);
		foreach (var method in matchingMethods)
			if (IsMethodWithMatchingParametersType(method, typesOfArguments, commonTypeOfArguments, Type) ||
				commonTypeOfArguments != null && commonTypeOfArguments ==
				GetListElementTypeIfHasSingleParameter(method, arguments.Count))
				return method;
		// Single character text can always be used as a character (thus number)
		if (arguments.Count == 1 && matchingMethods.Count > 0 &&
			matchingMethods[0].Parameters.Count > 0 &&
			(matchingMethods[0].Parameters[0].Type.IsNumber ||
				matchingMethods[0].Parameters[0].Type.IsCharacter) &&
			arguments[0].ReturnType.IsText && arguments[0].IsConstant &&
			arguments[0].GetType().Name == Text && GetTextValue(arguments[0]).Length == 1)
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
		throw new ArgumentsDoNotMatchMethodParameters(arguments, Type, matchingMethods);
	}

	private static string GetTextValue(Expression argument)
	{
		var data = argument.GetType().
			GetProperty("Data", BindingFlags.Instance | BindingFlags.Public)?.GetValue(argument);
		if (data is string value)
			return value; //ncrunch: no coverage
		var text = data?.ToString() ?? argument.ToString();
		const string ValueInstanceTextPrefix = "Text: \"";
		if (text.StartsWith(ValueInstanceTextPrefix, StringComparison.Ordinal) && text.EndsWith("\"", StringComparison.Ordinal))
			return text[ValueInstanceTextPrefix.Length..^1];
		return text.Length >= 2 && text[0] == '"' && text[^1] == '"' //ncrunch: no coverage
			? text[1..^1]
			: text;
	}

	private static T? TryGetSingleElementType<T>(IReadOnlyList<T> argumentTypes) where T : class
	{
		T? firstType = null;
		for (var i = 0; i < argumentTypes.Count; i++)
			if (firstType == null)
				firstType = argumentTypes[i];
			else if (firstType != argumentTypes[i])
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
					Generic.Name: List
				} parameterType
			}
		] && IsFromConstructorWithMatchingConstraints(method, numberOfArguments)
			? parameterType.ImplementationTypes[0]
			: null;

	private static bool IsFromConstructorWithMatchingConstraints(Method method, int numberOfArguments)
	{
		if (method.Name != Method.From)
			return true;
		var member = method.Type.Members.FirstOrDefault(m => !m.IsConstant && m.Type.Name != Iterator);
		return member?.Constraints == null ||
			ConstraintCouldBeSatisfiedByArgumentCount(member.Constraints[0], numberOfArguments);
	}

	private static bool ConstraintCouldBeSatisfiedByArgumentCount(Expression constraint, int numberOfArguments)
	{
		var constraintText = constraint.ToString();
		// Check for "Length is N" pattern
		if (constraintText.Contains("Length is "))
			return constraintText.Contains("Length is " + numberOfArguments);
		// Check for "Length > N", "Length >= N", "Length < N", "Length <= N" patterns
		//ncrunch: no coverage start
		if (constraintText.Contains("Length"))
		{
			if (constraintText.Contains("> 0"))
				return numberOfArguments > 0;
			if (constraintText.Contains(">= "))
				return TryExtractNumberAndCompare(constraintText, ">=", numberOfArguments);
			if (constraintText.Contains("> "))
				return TryExtractNumberAndCompare(constraintText, ">", numberOfArguments);
			if (constraintText.Contains("< "))
				return TryExtractNumberAndCompare(constraintText, "<", numberOfArguments);
			if (constraintText.Contains("<= "))
				return TryExtractNumberAndCompare(constraintText, "<=", numberOfArguments);
		}
		// For other constraint types (like " " is not in value), we assume they could pass
		return true;
	}

	private static bool TryExtractNumberAndCompare(string constraintText, string op, int numberOfArguments)
	{
		var opIndex = constraintText.IndexOf(op, StringComparison.Ordinal);
		if (opIndex < 0)
			return true;
		var afterOp = constraintText[(opIndex + op.Length)..].TrimStart();
		var numberEndIndex = 0;
		while (numberEndIndex < afterOp.Length && char.IsDigit(afterOp[numberEndIndex]))
			numberEndIndex++;
		if (numberEndIndex == 0)
			return true;
		if (!int.TryParse(afterOp[..numberEndIndex], out var constraintNumber))
			return true;
		return op switch
		{
			">" => numberOfArguments > constraintNumber,
			">=" => numberOfArguments >= constraintNumber,
			"<" => numberOfArguments < constraintNumber,
			"<=" => numberOfArguments <= constraintNumber,
			_ => true
		};
	} //ncrunch: no coverage end

	private static bool IsMethodWithMatchingParametersType(Method method,
		IReadOnlyList<Type> typesOfArguments, Type? commonTypeOfArguments, Type currentType)
	{
		// Allow `is`/`is not` comparisons against our own type (Range is Range), those are mostly not
		// implemented. Also, always allow comparison against Errors for error checking.
		if (method.Name is BinaryOperator.Is or UnaryOperator.Not && method.Parameters.Count > 0 &&
			method.Parameters[0].Type.IsAny && (commonTypeOfArguments == currentType ||
				(commonTypeOfArguments?.IsError ?? false)))
			return true; //ncrunch: no coverage
		if (method is { Name: Method.From, Parameters.Count: 0 } && typesOfArguments.Count == 1 &&
			method.ReturnType.IsSameOrCanBeUsedAs(typesOfArguments[0], false))
			return true; //ncrunch: no coverage
		if (typesOfArguments.Count > method.Parameters.Count || typesOfArguments.Count <
			GetMethodParameterDefaultValueCount(method))
			return false;
		for (var index = 0; index < typesOfArguments.Count; index++)
			if (!IsMethodParameterMatchingArgument(method, index, typesOfArguments[index]))
				return false;
		return true;
	}

	private static int GetMethodParameterDefaultValueCount(Method method)
	{
		var count = 0;
		for (var index = 0; index < method.Parameters.Count; index++)
			if (method.Parameters[index].DefaultValue == null &&
				!CanAutoCreateType(method.Parameters[index].Type))
				count++;
		return count;
	}

	/// <summary>
	/// A type can be auto-created (injected without explicit argument) if it is a trait
	/// (resolved via the runtime's trait-implementation registry) or if it is a concrete type
	/// all of whose members are themselves auto-creatable (e.g. Logger whose only member is
	/// the TextWriter trait).
	/// </summary>
	private static bool CanAutoCreateType(Type type, HashSet<Type>? visiting = null)
	{
		if (type.IsTrait)
			return true;
		if (type.IsNumber || type.IsBoolean || type.IsCharacter || type.IsText || type.IsNone ||
		    type.Members.Count == 0)
			return false;
		visiting ??= [];
		if (!visiting.Add(type))
			return false;
		var result = type.Members.All(m => CanAutoCreateType(m.Type, visiting));
		visiting.Remove(type);
		return result;
	}

	private static bool IsMethodParameterMatchingArgument(Method method, int index, Type argumentType)
	{
		var methodParameterType = method.Parameters[index].Type;
		if (methodParameterType.IsMutable)
			methodParameterType = methodParameterType.GetFirstImplementation();
		if (argumentType == methodParameterType || method.IsGeneric ||
			IsArgumentImplementationTypeMatchParameterType(argumentType, methodParameterType))
			return true;
		if (methodParameterType is { IsText: false, IsEnum: true } &&
			methodParameterType.Members[0].Type.IsSameOrCanBeUsedAs(argumentType))
			return true;
		if (methodParameterType.Name == Type.Iterator && method.Type.IsSameOrCanBeUsedAs(argumentType))
			return true; //ncrunch: no coverage
		if (!methodParameterType.IsGeneric)
			return argumentType.IsSameOrCanBeUsedAs(methodParameterType);
		if (argumentType.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(methodParameterType, //ncrunch: no coverage
				"(parameter " + index + ") is not usable with argument " + argumentType + " in " + method);
		return false;
	}

	private static bool
		IsArgumentImplementationTypeMatchParameterType(Type argumentType, Type parameterType) =>
		argumentType is GenericTypeImplementation argumentGenericType &&
		argumentGenericType.ImplementationTypes.Any(t => t == parameterType);
}