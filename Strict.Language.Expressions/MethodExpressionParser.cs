using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	private Stack<string> orderedTokens = new();

	public override Expression ParseAssignmentExpression(Type type, string initializationLine, int fileLineNumber)
	{
		var constructor = type.Methods[0];
		var line = new Method.Line(constructor, 0, initializationLine, fileLineNumber);
		return new MethodCall(new Value(type, type), constructor,
			TryParseExpression(line, initializationLine) ?? throw new UnknownExpression(line));
	}

	public override Expression? TryParseExpression(Method.Line line, ReadOnlySpan<char> partToParse)
	{
		if (partToParse.Contains(' '))
		{
			var postfixTokens = new ShuntingYard(partToParse).Output;
			if (postfixTokens.Count == 1)
			{
				var token = postfixTokens.Pop();
				return Text.TryParse(line, token) ?? List.TryParse(line, token);
			}
			if (postfixTokens.Count == 2)
			{
				// TODO: Unary goes here
			}
			return Binary.TryParse(line, postfixTokens);
		}
		return Text.TryParse(line, partToParse) ?? Boolean.TryParse(line.Method, partToParse) ??
			Number.TryParse(line.Method, partToParse);
		//TODO: Do below expression parsing outside of this method
		//?? Constructor.TryParse(line, partToParse) ??
		//MemberCall.TryParse(line, partToParse) ?? MethodCall.TryParse(line, partToParse);
	}

	public sealed class UnknownExpression : ParsingFailed
	{
		public UnknownExpression(Method.Line line, string error = "") : base(line, error) { }
	}

	/// <summary>
	/// Called lazily by Method.Body and only if needed for execution (context should be over there
	/// as parsing is done in parallel, we should not keep any state here).
	/// </summary>
	public override Expression ParseMethodBody(Method method)
	{
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < method.bodyLines.Count; lineNumber++)
		{
			var expression = ParseMethodLine(method.bodyLines[lineNumber], ref lineNumber);
			if (expression is Assignment assignment)
				method.Variables.Add(assignment);
			expressions.Add(expression);
		}
		//TODO: to clear link to original memory: method.bodyLines = Memory<char>.Empty; //ArraySegment<Method.Line>.Empty;
		return new MethodBody(method, expressions);
	}

	//private readonly PhraseTokenizer tokenizer = new();

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber)
	{
		try
		{
			return Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
				//TODO: for loop
				Return.TryParse(line) ??
				//TODO: error
				TryParseExpression(line, line.Text) ?? throw new UnknownExpression(line);
		}
		catch (UnknownExpression e)
		{
			return line.Groups.Count > 0
				? GroupExpressionParser.TryParse(line, 0)
				: throw e;
		}
	}
}