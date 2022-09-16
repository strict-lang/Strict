using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Strict.Language.Expressions;

/// <summary>
///   Parses method bodies by splitting into main lines (lines starting without tabs)
///   and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	/// <summary>
	///   Slightly slower version that checks high level expressions that can only occur at the line
	///   level like let, if, for (those will increase methodLineNumber as well) and return.
	///   Every other expression can be nested and can appear anywhere.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public override Expression ParseLineExpression(Body body, ReadOnlySpan<char> line) =>
		Assignment.TryParse(body, line) ?? If.TryParse(body, line) ??
		For.TryParse(body, line.Trim()) ?? Return.TryParse(body, line) ?? Mutable.TryParse(body, line) ?? ParseExpression(body, line);

	// ReSharper disable once CyclomaticComplexity
	// ReSharper disable once MethodTooLong
	public override Expression ParseExpression(Body body, ReadOnlySpan<char> input)
	{
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(body);
		if (input.Length < 3 || !input.Contains(' ') && !input.Contains(','))
			return Boolean.TryParse(body, input) ?? Text.TryParse(body, input) ??
				List.TryParseWithSingleElement(body, input) ?? Number.TryParse(body, input) ??
				TryParseMemberOrZeroOrOneArgumentMethodCall(body, input) ?? (input.IsOperator()
					? throw new InvalidOperatorHere(body, input.ToString())
					: input.IsWord()
						? throw new IdentifierNotFound(body, input.ToString())
						: throw new UnknownExpression(body, input.ToString()));
		if (input.StartsWith("error "))
			return TryParseErrorExpression(body, input[6..]);
		// If this is just a simple text string, there is no need to invoke ShuntingYard
		if (input[0] == '"' && input[^1] == '"' && input.Count('"') == 2)
			return new Text(body.Method, input.Slice(1, input.Length - 2).ToString());
		// If this is just a simple list, no need to invoke ShuntingYard yet, grab each list element
		if (input[0] == '(' && input[^1] == ')' && input.Contains(',') && input.Count('(') == 1)
			return new List(body, body.Method.ParseListArguments(body, input[1..^1]));
		// Conditionals are only supported here and can't be nested
		if (If.CanTryParseConditional(body, input))
			return If.ParseConditional(body, input);
		var postfix = new ShuntingYard(input.ToString());
		if (postfix.Output.Count == 1)
			return TryParseMemberOrZeroOrOneArgumentMethodCall(body, input) ??
				ParseTextWithSpacesOrListWithMultipleOrNestedElements(body, input[postfix.Output.Pop()]);
		if (postfix.Output.Count == 2)
			return ParseMethodCallWithArguments(body, input, postfix);
		var binary = Binary.Parse(body, input, postfix.Output);
		if (postfix.Output.Count == 0)
			return binary;
		return ParseInContext(body.Method.Type, body, input[postfix.Output.Peek()],
				new[] { binary }) ??
			throw new UnknownExpression(body, input[postfix.Output.Peek()].ToString());
	}

	private Error TryParseErrorExpression(Body body, ReadOnlySpan<char> partToParse)
	{
		var expression = ParseExpression(body, partToParse);
		if (expression.ReturnType.Name != Base.Text)
			throw new ArgumentException("Error must be a text");
		return new Error(expression);
	}

	private Expression ParseMethodCallWithArguments(Body body, ReadOnlySpan<char> input,
		ShuntingYard postfix)
	{
		var argumentsRange = postfix.Output.Pop();
		var methodRange = postfix.Output.Pop();
#if LOG_DETAILS
		Logger.Info(nameof(ParseMethodCallWithArguments) + ", method=" +
			input[methodRange].ToString() + " arguments=" + input[argumentsRange].ToString());
#endif
		if (input[argumentsRange.Start.Value] == '(')
			return ParseInContext(body.Method.Type, body,
					input[methodRange], // MethodCall always produces single token from ShutingYard so this call never happens atm, I think using unary operator could be a way to hit this line
					ParseListArguments(body,
						input[(argumentsRange.Start.Value + 1)..(argumentsRange.End.Value - 1)])) ??
				throw new MemberOrMethodNotFound(body, body.Method.Type, input[methodRange].ToString());
		if (input[argumentsRange.Start.Value] == '.')
			return ParseInContext(body.Method.Type, body, input, Array.Empty<Expression>()) ??
				throw new InvalidOperatorHere(body, input[methodRange].ToString());
		return input[methodRange].Equals(UnaryOperator.Not, StringComparison.Ordinal)
			? new Not(body.Method.ParseExpression(body, input[argumentsRange]))
			: throw new InvalidOperatorHere(body, input[methodRange].ToString());
	}

	/// <summary>
	///   By far the most common usecase, we call something from another instance, use some binary
	///   operator (like is, to, +, etc.) or execute some method. For more arguments more complex
	///   parsing has to be done and we have to invoke ShuntingYard for the argument list.
	/// </summary>
	public Expression? TryParseMemberOrZeroOrOneArgumentMethodCall(Body body,
		ReadOnlySpan<char> input)
	{
		var argumentsStart = input.IndexOf('(');
		if (argumentsStart > 0 && input[^1] == ')')
			return ParseInContext(body.Method.Type, body, input[..argumentsStart],
				ParseListArguments(body, input[(argumentsStart + 1)..^1]));
		return ParseInContext(body.Method.Type, body, input, Array.Empty<Expression>());
	}

	// ReSharper disable once TooManyArguments
	// ReSharper disable once MethodTooLong
	private Expression? ParseInContext(Context context, Body body, ReadOnlySpan<char> input,
		IReadOnlyList<Expression> arguments)
	{
#if LOG_DETAILS
		Logger.Info(nameof(ParseInContext) + " " + context + ", " + input.ToString());
#endif
		if (input.Contains('.'))
		{
			var members = new RangeEnumerator(input, '.', 0);
			Expression? current = null;
			while (members.MoveNext())
			{
				if (current == null)
				{
					current = Text.TryParse(body, input[members.Current]) ??
						List.TryParseWithSingleElement(body, input[members.Current]);
					if (current != null)
					{
						context = current.ReturnType;
						continue;
					}
				}
				var expression = input[members.Current].Contains('(')
					? TryParseMemberOrZeroOrOneArgumentMethodCall(body, input[members.Current])
					: TryMemberOrMethodCall(context, current, body, input[members.Current],
						// arguments are only needed for the last part
						members.IsAtEnd
							? arguments
							: Array.Empty<Expression>());
				if (expression == null)
				{
					if (input[members.Current].IsOperator())
						throw new InvalidOperatorHere(body, input[members.Current].ToString());
					if (input[members.Current].TryParseNumber(out _))
						throw new NumbersCanNotBeInNestedCalls(body, input[members.Current].ToString());
					throw new MemberOrMethodNotFound(body, current?.ReturnType ?? body.Method.Type,
						input[members.Current].ToString());
				}
				current = expression;
				context = current.ReturnType;
			}
			return current;
		}
		return TryMemberOrMethodCall(context, null, body, input, arguments);
	}

	// ReSharper disable once TooManyArguments
	// ReSharper disable once ExcessiveIndentation
	// ReSharper disable once MethodTooLong
	private static Expression? TryMemberOrMethodCall(Context context, Expression? instance,
		Body body, ReadOnlySpan<char> input, IReadOnlyList<Expression> arguments)
	{
		if (!input.IsWord() && !input.Contains(' ') && !input.Contains('('))
			return null;
#if LOG_DETAILS
		Logger.Info(nameof(TryMemberOrMethodCall) + ": " + input.ToString() + " in " + context +
			" with arguments=" + arguments.ToWordList());
#endif
		var type = context as Type ?? body.Method.Type;
		if (arguments.Count == 0)
		{
			var variableValue = body.FindVariableValue(input);
			if (variableValue != null)
				return new VariableCall(input.ToString(), variableValue);
			foreach (var parameter in body.Method.Parameters)
				if (input.Equals(parameter.Name, StringComparison.Ordinal))
					return new ParameterCall(parameter);
			var memberCall = TryFindMemberCall(type, instance, input);
			if (memberCall != null)
				return memberCall;
#if LOG_DETAILS
			Logger.Info(nameof(TryMemberOrMethodCall) + " found no member in " + body.Method);
#endif
		}
		var methodName = input.ToString();
		var method2 = type.FindMethod(methodName, arguments);
		if (method2 != null)
			return new MethodCall(method2, instance, arguments);
		if (instance == null)
		{
			var fromType = body.Method.FindType(methodName);
			if (fromType != null)
				return new MethodCall(fromType.GetMethod(Method.From, arguments), new From(fromType),
					arguments);
		}
#if LOG_DETAILS
		Logger.Info("ParseNested found no local method in " + body.Method.Type + ": " + methodName);
#endif
		return null;
	}

	private static Expression? TryFindMemberCall(Type type, Expression? instance,
		ReadOnlySpan<char> partToParse)
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

	/// <summary>
	///   Figures out if there are any bracket groups or if there is binary expression action going on.
	///   Could also contain strings, we don't know. Most of the time it will just be a bunch of values.
	///   <see cref="ShuntingYard" /> will only parse till the next comma, has to call this till the end.
	/// </summary>
	// ReSharper disable once CyclomaticComplexity
	// ReSharper disable once ExcessiveIndentation
	// ReSharper disable once MethodTooLong
	public override List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> innerSpan)
	{
		if (innerSpan.Contains('(') || innerSpan.Contains('"') && innerSpan.Contains(' '))
		{
			if (If.CanTryParseConditional(body, innerSpan))
				return new List<Expression> { If.ParseConditional(body, innerSpan) };
			// The postfix data comes in upside down, so use another stack to restore order
			var expressions = new Stack<Expression>();
			// Similar to TryParseExpression, but we know there is commas separating things!
			var postfix = new ShuntingYard(innerSpan.ToString());
			if (postfix.Output.Count == 1)
				expressions.Push(ParseTextWithSpacesOrListWithMultipleOrNestedElements(body,
					innerSpan[postfix.Output.Pop()]));
			else if (postfix.Output.Count == 2)
				expressions.Push(
					ParseMethodCallWithArguments(body, innerSpan,
						postfix)); // this line could be tested after unary operator (e.g. not) is working
			else
				do
				{
#if LOG_DETAILS
					Logger.Info("pushing list element " + innerSpan[postfix.Output.Peek()].ToString());
#endif
					var span = innerSpan[postfix.Output.Peek()];
					// Is this a binary expression we have to put into the list (already tokenized and postfixed)
					try
					{
						if (span.Length == 1 && span[0].IsSingleCharacterOperator() ||
							span.IsMultiCharacterOperator())
							expressions.Push(Binary.Parse(body, innerSpan, postfix.Output));
						else
							expressions.Push(
								body.Method.ParseExpression(body, innerSpan[postfix.Output.Pop()]));
					}
					catch (UnknownExpression ex)
					{
						throw new UnknownExpressionForArgument(body,
							span.ToString() + " is invalid for argument " + expressions.Count + " " +
							ex.Message);
					}
					if (postfix.Output.Count > 0 && innerSpan[postfix.Output.Pop().Start.Value] != ',')
						throw new ListTokensAreNotSeparatedByComma(body);
				} while (postfix.Output.Count > 0);
			return new List<Expression>(expressions);
		}
		if (innerSpan.Length == 0)
			throw new List.EmptyListNotAllowed(body);
		return ParseAllElementsFast(body, innerSpan, new RangeEnumerator(innerSpan, ',', 0));
	}

	private static List<Expression> ParseAllElementsFast(Body body, ReadOnlySpan<char> input,
		RangeEnumerator elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
			try
			{
				expressions.Add(body.Method.ParseExpression(body, input[element]));
			}
			catch (UnknownExpression ex)
			{
				throw new UnknownExpressionForArgument(body,
					input[element].ToString() + " (argument " + expressions.Count + ")\n" + ex.StackTrace);
			}
#if LOG_DETAILS
		Logger.Info(nameof(ParseAllElementsFast) + ": " + expressions.ToWordList());
#endif
		return expressions;
	}

	private static Expression
		ParseTextWithSpacesOrListWithMultipleOrNestedElements(Body body, ReadOnlySpan<char> input) =>
		Text.TryParse(body, input) ?? List.TryParseWithMultipleOrNestedElements(body, input) ??
		throw new InvalidSingleTokenExpression(body, input.ToString());

	public sealed class InvalidOperatorHere : ParsingFailed
	{
		public InvalidOperatorHere(Body body, string message) : base(body, message) { }
	}

	public sealed class IdentifierNotFound : ParsingFailed
	{
		public IdentifierNotFound(Body body, string name) : base(body, name) { }
	}

	public sealed class UnknownExpression : ParsingFailed
	{
		public UnknownExpression(Body body, string error = "") : base(body, error) { }
	}

	public class CannotParseEmptyInput : ParsingFailed
	{
		public CannotParseEmptyInput(Body body) : base(body) { }
	}

	public sealed class NumbersCanNotBeInNestedCalls : ParsingFailed
	{
		public NumbersCanNotBeInNestedCalls(Body body, string text) : base(body, text) { }
	}

	public sealed class MemberOrMethodNotFound : ParsingFailed
	{
		public MemberOrMethodNotFound(Body body, Type memberType, string memberName) : base(body,
			memberName, memberType) { }
	}

	public sealed class UnknownExpressionForArgument : ParsingFailed
	{
		public UnknownExpressionForArgument(Body body, string message) : base(body, message) { }
	}

	public class ListTokensAreNotSeparatedByComma : ParsingFailed
	{
		public ListTokensAreNotSeparatedByComma(Body body) : base(body) { }
	}

	private sealed class InvalidSingleTokenExpression : ParsingFailed
	{
		public InvalidSingleTokenExpression(Body body, string message) :
			base(body, message) { }
	}
}