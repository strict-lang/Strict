using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public class OneArgumentMethodCall : NoArgumentMethodCall
{
	public OneArgumentMethodCall(Method method, Expression? instance, Expression argument) :
		base(method, instance) =>
		Argument = argument;

	public Expression Argument { get; }
	public override string ToString() => base.ToString() + "(" + Argument + ")";
}

/// <summary>
/// Almost all method calls use zero, one or two arguments, the maximum is three arguments.
/// </summary>
// ReSharper disable once HollowTypeName
public class ArgumentsMethodCall : NoArgumentMethodCall
{
	public ArgumentsMethodCall(Method method, Expression? instance, Expression[] arguments) :
		base(method, instance) =>
		Arguments = arguments;

	public Expression[] Arguments { get; }
	public override string ToString() => base.ToString() + Arguments.ToBrackets();
}

/// <summary>
/// Any type of method we can call, this includes normal local method calls, recursions, calls to
/// any of our implement base types (instance is null in all of those cases), calls to other types
/// (either From(type) or instance method calls, there are no static methods) or any operator
/// <see cref="Binary"/> or <see cref="Not"/> unary call (which are all normal methods as well).
/// </summary>
// ReSharper disable once HollowTypeName
public class NoArgumentMethodCall : Expression
{
	public NoArgumentMethodCall(Method method, Expression? instance = null) : base(method.ReturnType)
	{
		Instance = instance;
		Method = method;
	}

	public Method Method { get; }
	public Expression? Instance { get; }

	public override string ToString() =>
		(Instance != null
			? Instance + "."
			: "") + Method.Name;
}