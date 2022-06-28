using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public class Constructor : Expression
{
	public Constructor(Method method) : base(
		method.ReturnType) =>
		Method = method;

	public Method Method { get; }

	public static Expression? TryParse(Method.Line line, string partToParse) =>
		partToParse.EndsWith(')') && partToParse.Contains('(')
			? TryParseConstructor(line,
				partToParse.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries))
			: null;

	private static Expression? TryParseConstructor(Method.Line line, IReadOnlyList<string> parts)
	{
		if (parts.Count >= 2)
		{
			try
			{
				var type = line.Method.GetType(parts[0]);
				var constructor = type.Methods[0];
				return new Assignment(new Identifier(parts[0], type), new MethodCall(new Value(type, type), constructor,
					line.Method.TryParseExpression(line, parts[1]) ?? throw new MethodExpressionParser.UnknownExpression(line)));
			}
			catch (Context.TypeNotFound)
			{
				return null;
			}
			//return new Assignment(new Identifier(parts[0], value.ReturnType), value);
		}
		return null;
	}
}