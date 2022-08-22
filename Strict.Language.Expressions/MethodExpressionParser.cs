using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	public override Expression ParseAssignmentExpression(Type type,
		ReadOnlySpan<char> initializationLine, int fileLineNumber)
	{
		var line = new Method.Line(type.Methods[0], 0, initializationLine.ToString(), fileLineNumber); // TODO: No idea how to avoid line here
		var arguments = new[] { ParseExpression(line, ..) };
		return new MethodCall(type.GetMethod(Method.From, arguments), new From(type), arguments);
	}

	public override Expression ParseExpression(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(line);
		if (input.Length < 3 || !input.Contains(' ') && !input.Contains(','))
			return Boolean.TryParse(line, range) ?? Text.TryParse(line, range) ??
				List.TryParseWithSingleElement(line, range) ?? Number.TryParse(line, range) ??
				TryParseMemberOrZeroOrOneArgumentMethodCall(line, range) ?? (input.IsOperator()
					? throw new InvalidOperatorHere(line, input.ToString())
					: input.IsWord()
						? throw new IdentifierNotFound(line, line.Text[range])
						: throw new UnknownExpression(line, line.Text[range]));
		// If this is just a simple text string, there is no need to invoke ShuntingYard
		if (input[0] == '"' && input[^1] == '"' && input.Count('"') == 2)
			return new Text(line.Method, input.Slice(1, input.Length - 2).ToString());
		// If this is just a simple list, no need to invoke ShuntingYard yet, grab each list element
		if (input[0] == '(' && input[^1] == ')' && input.Contains(',') && input.Count('(') == 1)
			return new List(line,
				line.Method.ParseListArguments(line, range.RemoveFirstAndLast(line.Text.Length)));
		// Conditionals are only supported here and can't be nested
		if (If.CanTryParseConditional(line, range))
			return If.ParseConditional(line, range);
		var postfix = new ShuntingYard(line.Text, range);
		if (postfix.Output.Count == 1)
			return TryParseMemberOrZeroOrOneArgumentMethodCall(line, range) ??
				ParseTextWithSpacesOrListWithMultipleOrNestedElements(line, postfix.Output.Pop());
		if (postfix.Output.Count == 2)
			return ParseMethodCallWithArguments(line, postfix);
		var binary = Binary.Parse(line, postfix.Output);
		if (postfix.Output.Count == 0)
			return binary;
		return ParseInContext(line.Method.Type, line, postfix.Output.Peek(), new[] { binary }) ??
			throw new UnknownExpression(line, line.Text[postfix.Output.Peek()]);
	}

	private Expression ParseMethodCallWithArguments(Method.Line line, ShuntingYard postfix)
	{
		var argumentsRange = postfix.Output.Pop();
		var methodRange = postfix.Output.Pop();
#if LOG_DETAILS
		Logger.Info(nameof(ParseMethodCallWithArguments) + ", method=" + line.Text[methodRange] +
			" arguments=" + line.Text[argumentsRange]);
#endif
		if (line.Text[argumentsRange.Start.Value] == '(')
			return ParseInContext(line.Method.Type, line, methodRange,
				ParseListArguments(line, argumentsRange.RemoveFirstAndLast(line.Text.Length))) ?? throw new MemberOrMethodNotFound(line, line.Method.Type, line.Text[methodRange]);
		return line.Text.GetSpanFromRange(methodRange).Equals(UnaryOperator.Not, StringComparison.Ordinal)
			? new Not(line.Method.ParseExpression(line, argumentsRange))
			: throw new InvalidOperatorHere(line, line.Text[methodRange]);
	}

	public sealed class InvalidOperatorHere : ParsingFailed
	{
		public InvalidOperatorHere(Method.Line line, string message) : base(line, message) { }
	}

	public sealed class IdentifierNotFound : ParsingFailed
	{
		public IdentifierNotFound(Method.Line line, string name) : base(line, name) { }
	}

	public sealed class UnknownExpression : ParsingFailed
	{
		public UnknownExpression(Method.Line line, string error = "") : base(line, error) { }
	}

	public class CannotParseEmptyInput : ParsingFailed
	{
		public CannotParseEmptyInput(Method.Line line) : base(line) { }
	}

	//https://deltaengine.fogbugz.com/f/cases/25211

	/// <summary>
	/// By far the most common usecase, we call something from another instance, use some binary
	/// operator (like is, to, +, etc.) or execute some method. For more arguments more complex
	/// parsing has to be done and we have to invoke ShuntingYard for the argument list.
	/// </summary>
	public Expression? TryParseMemberOrZeroOrOneArgumentMethodCall(Method.Line line, Range range)
	{
		var toParse = line.Text.GetSpanFromRange(range);
		var argumentsStart = toParse.IndexOf('(');
		if (argumentsStart > 0 && toParse[^1] == ')')
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
				var expression = line.Text.GetSpanFromRange(members.Current).Contains('(')
					? TryParseMemberOrZeroOrOneArgumentMethodCall(line, members.Current)
					: TryMemberOrMethodCall(context, current, line, members.Current,
						// arguments are only needed for the last part
						members.IsAtEnd
							? arguments
							: Array.Empty<Expression>());
				if (expression == null)
				{
					if (line.Text.GetSpanFromRange(members.Current).IsOperator())
						throw new InvalidOperatorHere(line, line.Text[members.Current]);
					if (line.Text.GetSpanFromRange(members.Current).TryParseNumber(out _))
						throw new NumbersCanNotBeInNestedCalls(line, line.Text[members.Current]);
					throw new MemberOrMethodNotFound(line, current?.ReturnType ?? line.Method.Type,
						line.Text[members.Current]);
				}
				current = expression;
				context = current.ReturnType;
			}
			return current;
		}
		return TryMemberOrMethodCall(context, null, line, range, arguments);
	}

	public sealed class NumbersCanNotBeInNestedCalls : ParsingFailed
	{
		public NumbersCanNotBeInNestedCalls(Method.Line line, string text) : base(line, text) { }
	}

	private static Expression? TryMemberOrMethodCall(Context context, Expression? instance, Method.Line line, Range range,
		IReadOnlyList<Expression> arguments)
	{
		var partToParse = line.Text.GetSpanFromRange(range);
		if (!partToParse.IsWord() && !partToParse.Contains(' ') && !partToParse.Contains('('))
			return null;
#if LOG_DETAILS
		Logger.Info(nameof(TryMemberOrMethodCall) + ": " + partToParse.ToString() + " in " + context +
			" with arguments=" + arguments.ToWordList());
#endif
		var type = context as Type ?? line.Method.Type;
		if (arguments.Count == 0)
		{
			var variableValue = line.Body?.FindVariableValue(partToParse);
			if (variableValue != null)
				return new VariableCall(partToParse.ToString(), variableValue);
			if (context is Method method)
			{
				foreach (var parameter in method.Parameters)
					if (partToParse.Equals(parameter.Name, StringComparison.Ordinal))
						return new ParameterCall(parameter);
				type = method.ReturnType;
			}
			var memberCall = TryFindMemberCall(type, instance, partToParse);
			if (memberCall != null)
				return memberCall;
#if LOG_DETAILS
			Logger.Info(nameof(TryMemberOrMethodCall) + " found no member in " + line.Method);
#endif
		}
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
		Logger.Info("ParseNested found no local method in " + line.Method.Type + ": " + methodName);
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
		if (innerSpan.Contains('(') || innerSpan.Contains('"') && innerSpan.Contains(' '))
		{
			if (If.CanTryParseConditional(line, range))
				return new List<Expression> { If.ParseConditional(line, range) };
			// The postfix data comes in upside down, so use another stack to restore order
			var expressions = new Stack<Expression>();
			// Similar to TryParseExpression, but we know there is commas separating things!
			var postfix = new ShuntingYard(line.Text, range);
			if (postfix.Output.Count == 1)
				expressions.Push(ParseTextWithSpacesOrListWithMultipleOrNestedElements(line,
					postfix.Output.Pop()));
			else if (postfix.Output.Count == 2)
				expressions.Push(ParseMethodCallWithArguments(line, postfix));
			else
				do
				{
#if LOG_DETAILS
					Logger.Info("pushing list element " + line.Text[postfix.Output.Peek()]);
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
						throw new UnknownExpressionForArgument(line,
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

	public sealed class UnknownExpressionForArgument : ParsingFailed
	{
		public UnknownExpressionForArgument(Method.Line line, string message) : base(line, message) { }
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
				throw new UnknownExpressionForArgument(line,
					line.Text[element] + " (argument " + expressions.Count + ")\n" + ex.StackTrace);
			}
#if LOG_DETAILS
		Logger.Info(nameof(ParseAllElementsFast) + ": " + expressions.ToWordList());
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

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		//https://deltaengine.fogbugz.com/f/cases/25210
		Return.TryParse(line) ?? ParseExpression(line, ..);
}