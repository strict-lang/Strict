using System;
using System.Collections.Generic;
using System.Linq;

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
		var partToParse = line.Text.GetSpanFromRange(range).ToString();//TODO!
		return partToParse.EndsWith(')') && partToParse.Contains('(')
			? TryParseMethod(line,
				//TODO: this should be list parsing
				partToParse.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries))
			//TODO: new should be avoided in all TryParse
			: TryParseMethod(line, new string[] { partToParse });
	}

	private static Expression? TryParseMethod(Method.Line line, params string[] parts)
	{
		var methodName = parts[0];
		var arguments = parts.Length > 1
			? GetArguments(line, parts[1], methodName)
			: Array.Empty<Expression>();
		if (!methodName.Contains('.'))
			return FindMethodCall(null, line, methodName, arguments);
		var memberParts = methodName.Split('.', 2);
		var firstMember = MemberCall.TryParse(line, ..);//TODO: bada bad badbada bdabda memberParts[0]);
		if (firstMember == null)
			throw new MemberCall.MemberNotFound(line, line.Method.Type, memberParts[0]);
		return FindMethodCall(firstMember, line, memberParts[1], arguments);
	}

	private static Expression[] GetArguments(Method.Line line, string argumentsText, string methodName)
	{
		// someClass.ComplicatedMethod((1, 2, 3) + (4, 5), 7)
		// list of 2 arguments:
		// [0] = (1, 2, 3) + (4, 5)
		// [1] = 7
		// don't use this, broken, we already have working list parsing
		var parts = argumentsText.Split(", ");
		var arguments = new Expression[parts.Length];
		for (var index = 0; index < parts.Length; index++)
			arguments[index] = line.Method.TryParseExpression(line, ..) ?? //TODO: broken, fix it! parts[index]) ??
				throw new InvalidExpressionForArgument(line,
					parts[index] + " for " + methodName + " argument " + index);
		return arguments;
	}

	public sealed class InvalidExpressionForArgument : ParsingFailed
	{
		public InvalidExpressionForArgument(Method.Line line, string message) : base(line, message) { }
	}

	// ReSharper disable once TooManyArguments
	private static MethodCall? FindMethodCall(Expression? instance, Method.Line line, string methodName,
		Expression[] arguments)
	{
		if (!methodName.IsWord())
			return null;
		var context = instance?.ReturnType ?? line.Method.Type;
		var method = context.Methods.FirstOrDefault(m => m.Name == methodName);
		return method == null
			? throw new MethodNotFound(line, methodName, context)
			: method.Parameters.Count != arguments.Length
				? throw new ArgumentsDoNotMatchMethodParameters(line, arguments, method)
				: new MethodCall(instance, method, arguments);
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
}