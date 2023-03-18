using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// Any type of method we can call, this includes normal local method calls, recursions, calls to
/// any of our implement base types (instance is null in all of those cases), calls to other types
/// (either From(type) or instance method calls, there are no static methods) or any operator
/// <see cref="Binary"/> or <see cref="Not"/> unary call (which are all normal methods as well).
/// Like MemberCall has the same syntax when parent instance is used: Type.Method
/// </summary>
public class MethodCall : ConcreteExpression
{
	public MethodCall(Method method, Expression? instance, IReadOnlyList<Expression> arguments,
		Type? toReturnType = null) :
		base(GetMethodReturnType(method, toReturnType))
	{
		if (method.Name == Method.From && instance != null)
			throw new NotSupportedException("Makes no sense, we don't have an instance yet"); //ncrunch: no coverage
		Instance = instance;
		Method = method;
		Arguments = arguments;
	}

	private static Type GetMethodReturnType(Method method, Type? toReturnType) =>
		method.Name == BinaryOperator.To && toReturnType != null
			? toReturnType
			: method.ReturnType;

	public Method Method { get; }
	public Expression? Instance { get; }
	public IReadOnlyList<Expression> Arguments { get; }

	public MethodCall(Method method, Expression? instance = null, Type? toReturnType = null) : this(method, instance,
		Array.Empty<Expression>(), toReturnType) { }

	public static Expression? TryParse(Expression? instance, Body body, IReadOnlyList<Expression> arguments,
		Type type, string inputAsString)
	{
		if (body.IsFakeBodyForMemberInitialization)
			return null;
		var method = type.FindMethod(inputAsString, arguments, body.Method.Parser);
		if (method != null)
			return new MethodCall(method, instance, AreArgumentsAutoParsedAsList(method, arguments)
				? new List<Expression> { new List(body, (List<Expression>)arguments) }
				: arguments);
		return null;
	}

	private static bool
		AreArgumentsAutoParsedAsList(Method method, IReadOnlyCollection<Expression> arguments) =>
		method.Parameters.Count != arguments.Count && method.Parameters.Count == 1 &&
		arguments.Count > 1;

	public static Expression? TryParseFromOrEnum(Body body, IReadOnlyList<Expression> arguments,
		string methodName)
	{
		var fromType = body.Method.FindType(methodName);
		return fromType == null
			? null
			: IsConstructorUsedWithSameArgumentType(arguments, fromType)
				? throw new ConstructorForSameTypeArgumentIsNotAllowed(body)
				: new MethodCall(fromType.GetMethod(Method.From, arguments, body.Method.Parser), null,
					arguments);
	}

	private static bool
		IsConstructorUsedWithSameArgumentType(IReadOnlyList<Expression> arguments, Type fromType) =>
		arguments.Count == 1 && (fromType == arguments[0].ReturnType ||
			arguments[0].ReturnType is GenericTypeImplementation genericType && fromType == genericType.Generic);

	public sealed class ConstructorForSameTypeArgumentIsNotAllowed : ParsingFailed
	{
		public ConstructorForSameTypeArgumentIsNotAllowed(Body body) : base(body) { }
	}

	public override string ToString() =>
		Instance != null
			? $"{Instance}.{Method.Name}{Arguments.ToBrackets()}"
			: $"{GetProperMethodName()}{Arguments.ToBrackets()}";

	private string GetProperMethodName() =>
		Method.Name == Method.From
			? Method.ReturnType.Name
			: Method.Name;
}