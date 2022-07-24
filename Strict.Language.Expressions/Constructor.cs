using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public class Constructor //TODO: merge with normal method call
{
	public static Expression? TryParse(Method.Line line, Range range)
	{
		var partToParse = line.Text[range];//TODO: use span here!
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
		var constructorMethodCall = new MethodCall(new Value(type, type), type.Methods[0],
			line.Method.TryParseExpression(line, ..) ??//TODO: broken anyways: typeNameAndArguments[1]) ??
			throw new MethodExpressionParser.UnknownExpression(line));
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
			arguments[index] = line.Method.TryParseExpression(line, ..) ??//TODO: parts[index]) ??
				throw new MethodCall.InvalidExpressionForArgument(line,
					parts[index] + " for " + parts[index] + " argument " + index);
		return arguments;
	}
}