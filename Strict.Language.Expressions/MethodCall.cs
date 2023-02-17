using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using static System.Runtime.InteropServices.JavaScript.JSType;

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
			return new MethodCall(method, instance, arguments);
#if LOG_DETAILS
		Logger.Info("ParseNested found no local method in " + body.Method.Type + ": " + inputAsString);
#endif
		return null;
	}

	public static Expression? TryParseFromOrEnum(Body body, IReadOnlyList<Expression> arguments,
		string methodName)
	{
		var fromType = body.Method.FindType(methodName);
		return fromType == null
			? null
			: IsConstructorUsedWithSameArgumentType(arguments, fromType)
				? throw new ConstructorForSameTypeArgumentIsNotAllowed(body)
				: TryParseDictionary(fromType, arguments) ?? new MethodCall(
					fromType.GetMethod(Method.From, arguments, body.Method.Parser), null, arguments);
	}

	private static bool
		IsConstructorUsedWithSameArgumentType(IReadOnlyList<Expression> arguments, Type fromType) =>
		arguments.Count == 1 && (fromType == arguments[0].ReturnType ||
			arguments[0].ReturnType is GenericTypeImplementation genericType && fromType == genericType.Generic);

	public sealed class ConstructorForSameTypeArgumentIsNotAllowed : ParsingFailed
	{
		public ConstructorForSameTypeArgumentIsNotAllowed(Body body) : base(body) { }
	}

	private static Expression? TryParseDictionary(Type type, IReadOnlyList<Expression> arguments)
	{
		if (type.Name == Base.Dictionary && arguments.Count == 2)
			return new Dictionary(type, arguments[0].ReturnType, arguments[1].ReturnType);
		return null;
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

public class Dictionary : Value
{
	public Dictionary(Type type, Type keyType, Type mappedValueType) : base(type,
		new Dictionary<Type, Type>())
	{
		Key = keyType;
		MappedValueType = mappedValueType;
	}

	private Type Key { get; }
	private Type MappedValueType { get; }
}