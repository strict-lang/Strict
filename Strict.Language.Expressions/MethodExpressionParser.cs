using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	public override Expression ParseAssignmentExpression(Type type, string initializationLine, int fileLineNumber)
	{
		var constructor = type.Methods[0];
		var line = new Method.Line(constructor, 0, initializationLine, fileLineNumber);
		return new MethodCall(constructor, new From(type)); //TODO: argument logic need some more tests, ParseExpression(line, ..)); maybe non constructor calls also make sense here?
	}

	public override Expression ParseExpression(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(line);
		if (input.Length < 3 || !input.Contains(' ') && !input.Contains(','))
			return Boolean.TryParse(line, range) ?? Text.TryParse(line, range) ??
				List.TryParseWithSingleElement(line, range) ??
				TryParseMemberOrZeroOrOneArgumentMethodCall(line, range) ??
				Number.TryParse(line, range) ?? (input.IsOperator()
					? throw new InvalidOperatorHere(line, input.ToString())
					: throw new UnknownExpression(line, line.Text[range]));
		// If this is just a simple text string, there is no need to invoke ShuntingYard
		if (input[0] == '"' && input[^1] == '"' && input.Count('"') == 2)
			return new Text(line.Method, input.Slice(1, input.Length - 2).ToString());
		var postfix = new ShuntingYard(line.Text, range);
		return postfix.Output.Count switch
		{
			1 => TryParseMemberOrZeroOrOneArgumentMethodCall(line, range) ?? ParseTextWithSpacesOrListWithMultipleOrNestedElements(line, postfix.Output.Pop()),
			//TODO: can also be any method call or anything we excluded above that was still 1 token
			2 => Not.Parse(line, postfix),
			_ => //TODO: should never happen here, Binary will complain if we have a comma there! postfix.Output.Count % 2 != 1 && line.Text[postfix.Output.Skip(1).First().Start.Value] == ',' ?
				Binary.Parse(line, postfix.Output)
		};
	}

	public sealed class InvalidOperatorHere : ParsingFailed
	{
		public InvalidOperatorHere(Method.Line line, string message) : base(line, message) { }
	}

	public sealed class UnknownExpression : ParsingFailed
	{
		public UnknownExpression(Method.Line line, string error = "") : base(line, error) { }
	}

	public class CannotParseEmptyInput : ParsingFailed
	{
		public CannotParseEmptyInput(Method.Line line) : base(line) { }
	}

	//TODO: error handling (same as constructor calling actually)
	//https://deltaengine.fogbugz.com/f/cases/25211

	/// <summary>
	/// By far the most common usecase, we call something from another instance, use some binary
	/// operator (like is, to, +, etc.) or execute some method. For more arguments more complex
	/// parsing has to be done and we have to invoke ShuntingYard for the argument list.
	/// </summary>
	public Expression? TryParseMemberOrZeroOrOneArgumentMethodCall(Method.Line line, Range range)
	{
		// We can early out here if this looks like a number digit
		if (char.IsNumber(line.Text[range.Start.Value]))
			return null;
		var toParse = line.Text.GetSpanFromRange(range);
		var argumentsStart = toParse.IndexOf('(');
		if (argumentsStart > 0)
			return ParseInContext(line.Method.Type, line,
				new Range(range.Start, range.Start.Value + argumentsStart),
				ParseListArguments(line,
					new Range(argumentsStart + range.Start.Value + 1, toParse.Length + range.Start.Value - 1)));
		return ParseInContext(line.Method.Type, line, range, Array.Empty<Expression>());
	}

	private Expression? ParseInContext(Context context, Method.Line line, Range range, IReadOnlyList<Expression> arguments)
	{
		var partToParse = line.Text.GetSpanFromRange(range);
#if LOG_DETAILS
		Logger.Info(nameof(ParseInContext) + " " + context + ", " + partToParse.ToString());
#endif
		if (partToParse.Contains('.'))
		{
			var members = new RangeEnumerator(partToParse, '.', range.Start);
			Expression? current = null;
			while (members.MoveNext())
			{
				if (current == null)
				{
					current = Text.TryParse(line, members.Current) ??
						List.TryParseWithSingleElement(line, members.Current);
					if (current != null)
					{
						context = current.ReturnType;
						continue;
					}
				}
				var expression = TryMemberOrMethodCall(context, current, line, members.Current,
					// arguments are only needed for the last part
					members.IsAtEnd
						? arguments
						: Array.Empty<Expression>());
				if (expression == null)
					throw new MemberOrMethodNotFound(line, current?.ReturnType ?? line.Method.Type,
						line.Text[members.Current]);
				current = expression;
				context = current.ReturnType;
			}
			return current;
		}
		return TryMemberOrMethodCall(context, null, line, range, arguments);
	}

	private static Expression? TryMemberOrMethodCall(Context context, Expression? instance, Method.Line line, Range range,
		IReadOnlyList<Expression> arguments)
	{
		var partToParse = line.Text.GetSpanFromRange(range);
		if (!partToParse.IsWord())
			return null;
#if LOG_DETAILS
		Logger.Info(nameof(TryMemberOrMethodCall) + ": " + partToParse.ToString()+" in "+context+" with arguments="+arguments.ToWordList());
#endif
		//foreach (var (name, variableValue) in GetAvailableVariables(context))
		//	if (partToParse.Equals(name, StringComparison.Ordinal))
		//		return variableValue;//TODO: should be member yo
		//TODO: test: Find all parent members as well use unit test -> Count(5).Floor is 5
		var type = context as Type ?? line.Method.Type;
		if (arguments.Count == 0)
		{
			if (context is Method method)
			{
				foreach (var (name, value) in method.Variables)
					if (partToParse.Equals(name, StringComparison.Ordinal))
						return new VariableCall(name, value);
				foreach (var parameter in method.Parameters)
					if (partToParse.Equals(parameter.Name, StringComparison.Ordinal))
						return new ParameterCall(parameter);
				type = method.ReturnType;
			}
			var memberCall = TryFindMemberCall(type!, instance, partToParse);
			if (memberCall != null)
				return memberCall;
#if LOG_DETAILS
			Logger.Info("ParseNested found no member in " + line.Method);
#endif
		}
		//TODO: the member can be anything, any expression, don't assume it is always a member/method!
		var methodName = partToParse.ToString();
		var method2 = type.FindMethod(methodName, arguments);
		if (method2 != null)
			return new MethodCall(method2, instance, arguments);
		if (instance == null)
		{
			var fromType = line.Method.FindType(methodName);
			if (fromType != null)
				return new MethodCall(fromType.GetMethod(Method.From, arguments), new From(fromType), arguments);
		}
#if LOG_DETAILS
		Logger.Info("ParseNested found no local method in " + line.Method.Type+": "+methodName);
#endif
		return null;
	}

	private static Expression? TryFindMemberCall(Type type, Expression? instance, ReadOnlySpan<char> partToParse)
	{
		foreach (var member in type.Members)
			if (partToParse.Equals(member.Name, StringComparison.Ordinal))
				return new MemberCall(instance, member);
		foreach (var implementType in type.Implements)
		{
			var memberCall = TryFindMemberCall(implementType, instance, partToParse);
			if (memberCall != null)
				return memberCall;
		}
		return null;
	}

	public sealed class MemberOrMethodNotFound : ParsingFailed
	{
		public MemberOrMethodNotFound(Method.Line line, Type memberType, string memberName) : base(line,
			memberName, memberType) { }
	}

	/// <summary>
	/// Figures out if there are any bracket groups or if there is binary expression action going on.
	/// Could also contain strings, we don't know. Most of the time it will just be a bunch of values.
	/// <see cref="ShuntingYard"/> will only parse till the next comma, has to call this till the end.
	/// </summary>
	public override List<Expression> ParseListArguments(Method.Line line, Range range)
	{
		var innerSpan = line.Text.GetSpanFromRange(range);
		if (innerSpan.Contains('(') || innerSpan.Contains('"'))
		{
			// The postfix data comes in upside down, so use another stack to restore order
			var expressions = new Stack<Expression>();
			// Similar to TryParseExpression, but we know there is commas separating things! 
			var postfix = new ShuntingYard(line.Text, range);
			if (postfix.Output.Count == 1)
				expressions.Push(ParseTextWithSpacesOrListWithMultipleOrNestedElements(line,
					postfix.Output.Pop()));
			else if (postfix.Output.Count == 2)
				expressions.Push(Not.Parse(line, postfix));
			else
				do
				{
#if LOG_DETAILS
					Logger.Info("pushing list element " +line.Text[postfix.Output.Peek()]);
#endif
					var span = line.Text.GetSpanFromRange(postfix.Output.Peek());
					// Is this a binary expression we have to put into the list (already tokenized and postfixed)
					try
					{
						if (span.Length == 1 && span[0].IsSingleCharacterOperator() ||
							span.IsMultiCharacterOperator())
							expressions.Push(Binary.Parse(line, postfix.Output));
						else
							expressions.Push(line.Method.ParseExpression(line, postfix.Output.Pop()));
					}
					catch (UnknownExpression ex)
					{
						throw new InvalidExpressionForArgument(line,
							span.ToString() + " is invalid for argument " + expressions.Count + " " +
							ex.Message);
					}
					if (postfix.Output.Count > 0 && line.Text[postfix.Output.Pop().Start.Value] != ',')
						throw new ListTokensAreNotSeparatedByComma(line);
				} while (postfix.Output.Count > 0);
			return new List<Expression>(expressions);
		}
		if (innerSpan.Length == 0)
			throw new List.EmptyListNotAllowed(line);
		return ParseAllElementsFast(line, new RangeEnumerator(innerSpan, ',', range.Start));
	}

	public sealed class InvalidExpressionForArgument : ParsingFailed
	{
		public InvalidExpressionForArgument(Method.Line line, string message) : base(line, message) { }
	}

	public class ListTokensAreNotSeparatedByComma : ParsingFailed
	{
		public ListTokensAreNotSeparatedByComma(Method.Line line) : base(line) { }
	}

	private static List<Expression> ParseAllElementsFast(Method.Line line, RangeEnumerator elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
			try
			{
				expressions.Add(line.Method.ParseExpression(line, element));
			}
			catch (UnknownExpression ex)
			{
				throw new InvalidExpressionForArgument(line,
					line.Text[element] + " is invalid for argument " + expressions.Count + " " + ex.Message);
			}
#if LOG_DETAILS
		Logger.Info(nameof(ParseAllElementsFast)+": "+expressions.ToWordList());
#endif
		return expressions;
	}

	private static Expression
		ParseTextWithSpacesOrListWithMultipleOrNestedElements(Method.Line line, Range range) =>
		Text.TryParse(line, range) ?? List.TryParseWithMultipleOrNestedElements(line, range) ??
		throw new InvalidSingleTokenExpression(line, line.Text[range]);

	private sealed class InvalidSingleTokenExpression : ParsingFailed
	{
		public InvalidSingleTokenExpression(Method.Line line, string message) : base(line, message) { }
	}

	/// <summary>
	/// Called lazily by Method.Body and only if needed for execution (context should be over there
	/// as parsing is done in parallel, we should not keep any state here).
	/// </summary>
	public override Expression ParseMethodBody(Method method)
	{
		if (method.bodyLines.Count == 0)
			return new MethodBody(method, Array.Empty<Expression>());
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < method.bodyLines.Count; lineNumber++)
			expressions.Add(ParseMethodLine(method.bodyLines[lineNumber], ref lineNumber));
		return new MethodBody(method, expressions);
	}

	//private readonly PhraseTokenizer tokenizer = new();

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		//https://deltaengine.fogbugz.com/f/cases/25210
		Return.TryParse(line) ?? ParseExpression(line, ..);
}