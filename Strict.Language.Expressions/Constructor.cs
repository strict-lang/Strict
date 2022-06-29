using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public class Constructor
{
	public static Expression? TryParse(Method.Line line, string partToParse) =>
		partToParse.EndsWith(')') && partToParse.Contains('(')
			? TryParseConstructor(line,
				partToParse.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries), partToParse.Contains(")."))
			: null;

	private static Expression? TryParseConstructor(Method.Line line,
		IReadOnlyList<string> parts, bool hasMethodCall)
	{
		var type = line.Method.FindType(parts[0]);
		if (type == null)
			return null;
		var constructorMethodCall = new MethodCall(new Value(type, type), type.Methods[0],
			line.Method.TryParseExpression(line, parts[1]) ??
			throw new MethodExpressionParser.UnknownExpression(line));
		if (!hasMethodCall)
			return new Assignment(new Identifier(parts[0], type),
				constructorMethodCall);
		var arguments = parts.Count > 3
			? GetArguments(line, parts.Skip(3).ToList())
			: Array.Empty<Expression>();
		var method = type.Methods.FirstOrDefault(m => m.Name == parts[2][1..]) ?? throw new MethodCall.MethodNotFound(line, parts[2][1..], type);
		return new MethodCall(constructorMethodCall, method, arguments);
	}

	private static Expression[] GetArguments(Method.Line line, IReadOnlyList<string> parts)
	{
		var arguments = new Expression[parts.Count];
		for (var index = 0; index < parts.Count; index++)
			arguments[index] = line.Method.TryParseExpression(line, parts[index]) ??
				throw new MethodCall.InvalidExpressionForArgument(line,
					parts[index] + " for " + parts[index] + " argument " + index);
		return arguments;
	}
}