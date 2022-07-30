using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public class MethodCall : Expression
{
	public MethodCall(Expression? instance, Method method, params Expression[] arguments) : base(
		method.ReturnType)
	{
		Instance = instance;
		Method = method;
		Arguments = arguments;
	}

	public Expression? Instance { get; }
	public Method Method { get; }
	public IReadOnlyList<Expression> Arguments { get; }

	public override string ToString() =>
		(Instance != null
			? Instance + "."
			: "") + Method.Name + Arguments.ToBrackets();

	public static Expression? TryParse(Method.Line line, Range range)
	{
		var partToParse = line.Text.GetSpanFromRange(range).ToString(); //TODO!
		return partToParse[^1] == ')' && partToParse.Contains('(')
			? TryParseMethod(line,
				//TODO: this should be list parsing
				range, partToParse.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries))
			//TODO: new should be avoided in all TryParse
			: TryParseMethod(line, range, new string[] { partToParse });
	}

	private static Expression? TryParseMethod(Method.Line line, Range range, params string[] parts)
	{
		var methodName = parts[0];
		var argumentStartIndex = range.Start.Value + parts[0].Length + 1;
		var arguments = parts.Length > 1
			? GetArguments(line, parts[1], methodName, argumentStartIndex)
			: Array.Empty<Expression>();
		if (!methodName.Contains('.'))
			// get the method
		{
			var method = FindMethod(null, line, methodName, arguments);
			if (method == null)
				//TODO: If not found check types for constructor call may be inline it here?
			{
				return
					TryParseConstructor(line,
						range); //range.Start.Value..(methodName.Length + range.Start.Value));
			}
			return new MethodCall(null, method, arguments);
		}
		var memberParts = methodName.Split('.', 2);
		var firstMember = MemberCall.TryParse(line, range.Start..memberParts[0].Length);
		if (firstMember == null)
			throw new MemberCall.MemberNotFound(line, line.Method.Type, memberParts[0]);
		var memberMethod = FindMethod(firstMember, line, memberParts[1], arguments);
		return memberMethod != null
			? new MethodCall(firstMember, memberMethod, arguments)
			: throw new MethodNotFound(line, memberParts[1], firstMember.ReturnType); // TODO: still check for members
	}

	private static Expression[] GetArguments(Method.Line line, string argumentsText,
		string methodName, int argumentStartIndex)
	{
		// someClass.ComplicatedMethod((1, 2, 3) + (4, 5), 7)
		// list of 2 arguments:
		// [0] = (1, 2, 3) + (4, 5)
		// [1] = 7
		// don't use this, broken, we already have working list parsing
		var parts = argumentsText.Split(", ");
		var arguments = new Expression[parts.Length];
		for (var index = 0; index < parts.Length; index++)
			try
			{
				arguments[index] = line.Method.ParseExpression(line,
					argumentStartIndex..(argumentStartIndex + parts[index].Length));
			}
			catch (MethodExpressionParser.UnknownExpression)
			{
				throw new InvalidExpressionForArgument(line, parts[index] + " is invalid for " + methodName + " argument " + index);
			}
		return arguments;
	}

	public sealed class InvalidExpressionForArgument : ParsingFailed
	{
		public InvalidExpressionForArgument(Method.Line line, string message) : base(line, message) { }
	}

	// ReSharper disable once TooManyArguments
	private static Method? FindMethod(Expression? instance, Method.Line line, string methodName,
		Expression[] arguments)
	{
		if (!methodName.IsWord())
			return null;
		var context = instance?.ReturnType ?? line.Method.Type;
		var method = context.Methods.FirstOrDefault(m => m.Name == methodName);
		return method; /* TODO: add once constructor is fixed == null
			? throw new MethodNotFound(line, methodName, context)
			: method.Parameters.Count != arguments.Length
				? throw new ArgumentsDoNotMatchMethodParameters(line, arguments, method)
				: method;*/
	}

	public sealed class MethodNotFound : ParsingFailed
	{
		public MethodNotFound(Method.Line line, string methodName, Type referencingType) : base(line, methodName, referencingType) { }
	}

	public sealed class ArgumentsDoNotMatchMethodParameters : ParsingFailed
	{
		public ArgumentsDoNotMatchMethodParameters(Method.Line line, Expression[] arguments,
			Method method) : base(line, (arguments.Length == 0
				? "No arguments does "
				: "Arguments: " + arguments.ToBrackets() + " do ") + "not match \"" + method.Type + "." +
			method.Name + "\" method parameters: " + method.Parameters.ToBrackets()) { }
	}

	public static Expression? TryParseConstructor(Method.Line line, Range range)
	{
		var partToParse = line.Text[range]; //TODO: use span here!
		return partToParse.EndsWith(')') && partToParse.Contains('(')
			? TryParseConstructor(line,
				partToParse.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries),
				partToParse.Contains(")."))
			: null;
	}

	private static Expression? TryParseConstructor(Method.Line line,
		IReadOnlyList<string> typeNameAndArguments, bool hasNestedMethodCall)
	{
		var type = line.Method.FindType(typeNameAndArguments[0]);
		if (type == null)
			return null;
		var constructorMethodCall = new MethodCall(new Value(type, type),
			type.Methods[0], // TODO: Get constructor method using a helper method
			line.Method.ParseExpression(line, ..)); //TODO: broken anyways: typeNameAndArguments[1]) ?? use same method GetArguments method
		if (!hasNestedMethodCall)
			return constructorMethodCall;
		var arguments = typeNameAndArguments.Count > 3
			? GetArguments(line, typeNameAndArguments.Skip(3).ToList())
			: Array.Empty<Expression>();
		var method = type.Methods.FirstOrDefault(m => m.Name == typeNameAndArguments[2][1..]) ?? throw new MethodCall.MethodNotFound(line, typeNameAndArguments[2][1..], type);
		return new MethodCall(constructorMethodCall, method, arguments);
	}

	private static Expression[] GetArguments(Method.Line line, IReadOnlyList<string> parts)
	{
		var arguments = new Expression[parts.Count];
		for (var index = 0; index < parts.Count; index++)
			//TODO: this is the same as above
			try
			{
				arguments[index] = line.Method.ParseExpression(line, ..); //TODO: parts[index]) ??
			}
			catch (MethodExpressionParser.UnknownExpression)
			{ //TODO: this is duplicated code!
				throw new InvalidExpressionForArgument(line,
					parts[index] + " is invalid for " + parts[index] + " argument " + index);
			}
		return arguments;
	}
}