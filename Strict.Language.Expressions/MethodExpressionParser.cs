using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	/// <summary>
	/// Slightly slower version that checks high level expressions that can only occur at the line
	/// level like let, if, for (those will increase methodLineNumber as well) and return.
	/// Every other expression can be nested and can appear anywhere.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public override Expression ParseLineExpression(Body body, ReadOnlySpan<char> line) =>
		ConstantDeclaration.TryParse(body, line, ConstantDeclaration.ConstantWithSpaceAtEnd) ?? If.TryParse(body, line) ??
		For.TryParse(body, line.Trim()) ?? Return.TryParse(body, line) ??
		ConstantDeclaration.TryParse(body, line, MutableDeclaration.MutableWithSpaceAtEnd) ??
		MutableAssignment.TryParse(body, line) ?? ParseExpression(body, line);

	public override Expression ParseExpression(Body body, ReadOnlySpan<char> input)
	{
		CheckIfEmptyOrAny(body, input);
		return input.Length < 3 || !input.Contains(' ') && !input.Contains(',')
			? TryParseCommon(body, input)
			: TryParseErrorOrTextOrListOrConditionalExpression(body, input) ??
			TryParseMethodOrMember(body, input);
	}

	private static void CheckIfEmptyOrAny(Body body, ReadOnlySpan<char> input)
	{
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(body);
		if (IsExpressionTypeAny(input))
			throw new ExpressionWithTypeAnyIsNotAllowed(body, input.ToString());
	}

	private static bool IsExpressionTypeAny(ReadOnlySpan<char> input) =>
		input.Equals(Base.Any, StringComparison.Ordinal) || input.StartsWith(Base.Any + "(");

	private Expression TryParseCommon(Body body, ReadOnlySpan<char> input) =>
		Boolean.TryParse(body, input) ?? Text.TryParse(body, input) ??
		List.TryParseWithSingleElement(body, input) ?? Number.TryParse(body, input) ??
		TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ?? (input.IsOperator()
			? throw new InvalidOperatorHere(body, input.ToString())
			: input.IsWord()
				? throw new Body.IdentifierNotFound(body, input.ToString())
				: throw new UnknownExpression(body, input.ToString()));

	private Expression? TryParseErrorOrTextOrListOrConditionalExpression(Body body, ReadOnlySpan<char> input) =>
		input.StartsWith("Error ")
			? TryParseErrorExpression(body, input[6..])
			:
			// If this is just a simple text string, there is no need to invoke ShuntingYard
			input[0] == '"' && input[^1] == '"' && input.Count('"') == 2
				? new Text(body.Method, input.Slice(1, input.Length - 2).ToString())
				:
				// If this is just a simple list, no need to invoke ShuntingYard yet, grab each list element
				input[0] == '(' && input[^1] == ')' && input.Contains(',') && input.Count('(') == 1
					? new List(body, body.Method.ParseListArguments(body, input[1..^1]))
					:
					// Conditionals are only supported here and can't be nested
					If.CanTryParseConditional(body, input)
						? If.ParseConditional(body, input)
						: null;

	private Error TryParseErrorExpression(Body body, ReadOnlySpan<char> partToParse)
	{
		var expression = ParseExpression(body, partToParse);
		if (expression.ReturnType.Name != Base.Text)
			throw new ArgumentException("Error must be a text but it is " + expression.ReturnType.Name);
		return new Error(expression);
	}

	private Expression TryParseMethodOrMember(Body body, ReadOnlySpan<char> input)
	{
		var postfix = new ShuntingYard(input.ToString());
		if (postfix.Output.Count == 1)
			return Dictionary.TryParse(body, input) ??
				TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ??
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

	private Expression ParseMethodCallWithArguments(Body body, ReadOnlySpan<char> input,
		ShuntingYard postfix)
	{
		var argumentsRange = postfix.Output.Pop();
		var methodRange = postfix.Output.Pop();
#if LOG_DETAILS
		Logger.Info(nameof(ParseMethodCallWithArguments) + ", method=" +
			input[methodRange].ToString() + " arguments=" + input[argumentsRange].ToString());
#endif
		return input[argumentsRange.Start.Value] == '('
			? ParseInContext(body.Method.Type, body, //ncrunch: no coverage
				input[methodRange],
				ParseListArguments(body,
					input[(argumentsRange.Start.Value + 1)..(argumentsRange.End.Value - 1)])) ??
			throw new MemberOrMethodNotFound(body, body.Method.Type, input[methodRange].ToString())
			: input[argumentsRange.Start.Value] == '.'
				? ParseInContext(body.Method.Type, body, input, Array.Empty<Expression>()) ??
				throw new InvalidOperatorHere(body, input[methodRange].ToString())
				: input[argumentsRange].Equals(UnaryOperator.Not, StringComparison.Ordinal)
					? Not.Parse(body, input, methodRange)
					: throw new InvalidOperatorHere(body, input[methodRange].ToString());
	}

	/// <summary>
	/// By far the most common usecase, we call something from another instance, use some binary
	/// operator (like is, to, +, etc.) or execute some method. For more arguments more complex
	/// parsing has to be done and we have to invoke ShuntingYard for the argument list.
	/// </summary>
	private Expression? TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(Body body,
		ReadOnlySpan<char> input)
	{
		var argumentsStart = input.IndexOf('(');
		var argumentsEnd = input.FindMatchingBracketIndex(argumentsStart);
		ChangeArgumentStartEndIfNestedMethodCall(input, ref argumentsStart, ref argumentsEnd);
		return argumentsStart <= 0 || argumentsEnd <= 0 || argumentsEnd < input.Length - 1
			? ParseInContext(body.Method.Type, body, input, Array.Empty<Expression>())
			: ParseInContext(body.Method.Type, body, input[..argumentsStart],
				ParseListArguments(body, input[(argumentsStart + 1)..argumentsEnd]));
	}

	private static void ChangeArgumentStartEndIfNestedMethodCall(ReadOnlySpan<char> input,
		ref int argumentsStart, ref int argumentsEnd)
	{
		if (IsNestedMethodCallWithParentMethodParameter(input, argumentsStart, argumentsEnd))
		{
			argumentsStart = input.LastIndexOf('(');
			argumentsEnd = input.FindMatchingBracketIndex(argumentsStart);
		}
	}

	private static bool IsNestedMethodCallWithParentMethodParameter(ReadOnlySpan<char> input,
		int argumentsStart, int argumentsEnd)
	{
		var innerArgumentStart = input.LastIndexOf('(');
		return argumentsStart != innerArgumentStart && argumentsEnd < innerArgumentStart &&
			input.IndexOf('.') < innerArgumentStart;
	}

	//https://deltaengine.fogbugz.com/f/cases/26383
	// ReSharper disable once TooManyArguments
	private Expression? ParseInContext(Context context, Body body, ReadOnlySpan<char> input,
		IReadOnlyList<Expression> arguments)
	{
#if LOG_DETAILS
		Logger.Info(nameof(ParseInContext) + " " + context + ", " + input.ToString());
#endif
		return input.Contains('.')
			? ParseNestedExpressionInContext(context, body, input, arguments)
			: ListCall.TryParse(body, TryVariableOrValueOrParameterOrMemberOrMethodCall(context, null, body, input, arguments),
				arguments);
	}

	private Expression? ParseNestedExpressionInContext(Context context, Body body, ReadOnlySpan<char> input,
		IReadOnlyList<Expression> arguments)
	{
		var members = new RangeEnumerator(input, '.', 0);
		Expression? current = null;
		while (members.MoveNext())
		{
			if (current == null)
			{
				current = Text.TryParse(body, input[members.Current]) ??
					List.TryParseWithMultipleOrNestedElements(body, input[members.Current]);
				if (current != null)
				{
					context = current.ReturnType;
					continue;
				}
			}
			var expression = input[members.Current].Contains('(')
				? TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input[members.Current])
				: TryVariableOrValueOrParameterOrMemberOrMethodCall(context, current, body, input[members.Current],
					// arguments are only needed for the last part
					members.IsAtEnd
						? arguments
						: Array.Empty<Expression>());
			// ReSharper disable once UnthrowableException
			current = expression ?? throw CheckErrorTypeAndThrowException(body, input, members, current);
			context = current.ReturnType;
		}
		return ListCall.TryParse(body, current, arguments);
	}

	private static Exception CheckErrorTypeAndThrowException(Body body, ReadOnlySpan<char> input,
		RangeEnumerator members, Expression? current) =>
		input[members.Current].IsOperator()
			? new InvalidOperatorHere(body, input[members.Current].ToString())
			: input[members.Current].TryParseNumber(out _)
				? new NumbersCanNotBeInNestedCalls(body, input[members.Current].ToString())
				: new MemberOrMethodNotFound(body, null, input[members.Current].ToString() +
					$" in {current?.ReturnType ?? body.Method.Type}" + (current?.ReturnType != null
						? ParsingFailed.GetClickableStacktraceLine(current.ReturnType, 0, string.Empty)
						: string.Empty));

	//https://deltaengine.fogbugz.com/f/cases/26383
	// ReSharper disable once TooManyArguments
	private static Expression? TryVariableOrValueOrParameterOrMemberOrMethodCall(Context context, Expression? instance,
		Body body, ReadOnlySpan<char> input, IReadOnlyList<Expression> arguments)
	{
		var inputAsString = input.ToString();
		var type = context as Type ?? body.Method.Type;
#if LOG_DETAILS
		Logger.Info(nameof(TryVariableOrValueOrParameterOrMemberOrMethodCall) + ": " + input.ToString() + " in " + context +
			" with arguments=" + arguments.ToWordList());
#endif
		return !input.IsWord() && !input.Contains(' ') && !input.Contains('(')
			? inputAsString.IsWordOrWordWithNumberAtEnd(out _)
				? MethodCall.TryParseFromOrEnum(body, arguments, inputAsString)
				: null
			: (VariableCall.TryParse(body, input) ?? (input.Equals(Base.Value, StringComparison.Ordinal)
				? Instance.Parse(body, body.Method)
				: ParameterCall.TryParse(body, input))) ?? (inputAsString.IsKeyword()
				? throw new KeywordNotAllowedAsMemberOrMethod(body, inputAsString, type)
				: (MemberCall.TryParse(body, type, instance, input) ??
					MethodCall.TryParse(instance, body, arguments, type, input.ToString())) ??
				(instance == null
					? MethodCall.TryParseFromOrEnum(body, arguments, inputAsString)
					: null));
	}

	public sealed class CannotAccessMemberBeforeTypeIsParsed : ParsingFailed
	{
		public CannotAccessMemberBeforeTypeIsParsed(Body body, string input, Type type) : base(body, input, type) { }
	}

	public sealed class KeywordNotAllowedAsMemberOrMethod : ParsingFailed
	{
		public KeywordNotAllowedAsMemberOrMethod(Body body, string input, Type type) : base(body,
			input, type) { }
	}

	/// <summary>
	/// Figures out if there are any bracket groups or if there is binary expression action going on.
	/// Could also contain strings, we don't know. Most of the time it will just be a bunch of values.
	/// <see cref="ShuntingYard" /> will only parse till the next comma, has to call this till the end.
	/// </summary>
	public override List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> innerSpan)
	{
		if (innerSpan.Contains('(') || innerSpan.Contains('"') && innerSpan.Contains(' '))
		{
			if (If.CanTryParseConditional(body, innerSpan))
				return new List<Expression> { If.ParseConditional(body, innerSpan) };
			// Similar to TryParseExpression, but we know there is commas separating things!
			var postfix = new ShuntingYard(innerSpan.ToString());
			// The postfix data comes in upside down, so use another stack to restore order
			var expressions = GetListArgumentsUsingPostfixTokens(body, innerSpan, postfix);
			return new List<Expression>(expressions);
		}
		if (innerSpan.Length == 0)
			throw new List.EmptyListNotAllowed(body);
		return ParseAllElementsFast(body, innerSpan, new RangeEnumerator(innerSpan, ',', 0));
	}

	private Stack<Expression> GetListArgumentsUsingPostfixTokens(Body body, ReadOnlySpan<char> innerSpan,
		ShuntingYard postfix)
	{
		var expressions = new Stack<Expression>();
		if (postfix.Output.Count == 1)
			expressions.Push(ParseTextWithSpacesOrListWithMultipleOrNestedElements(body,
				innerSpan[postfix.Output.Pop()]));
		else if (postfix.Output.Count == 2)
			expressions.Push(
				ParseMethodCallWithArguments(body, innerSpan,
					postfix));
		else
			ParseBinaryOrNormalExpressionsIntoList(body, innerSpan, postfix, expressions);
		return expressions;
	}

	private static void ParseBinaryOrNormalExpressionsIntoList(Body body, ReadOnlySpan<char> innerSpan, ShuntingYard postfix,
		Stack<Expression> expressions)
	{
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
					expressions.Push(body.Method.ParseExpression(body, innerSpan[postfix.Output.Pop()]));
			}
			catch (UnknownExpression ex)
			{
				throw new UnknownExpressionForArgument(body,
					span.ToString() + " is invalid for argument " + expressions.Count + " " + ex.Message);
			}
			if (postfix.Output.Count > 0 && innerSpan[postfix.Output.Pop().Start.Value] != ',')
				throw new ListTokensAreNotSeparatedByComma(body);
		} while (postfix.Output.Count > 0);
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

	private Expression
		ParseTextWithSpacesOrListWithMultipleOrNestedElements(Body body, ReadOnlySpan<char> input) =>
		Text.TryParse(body, input) ?? List.TryParseWithMultipleOrNestedElements(body, input) ??
		TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ??
		throw new InvalidSingleTokenExpression(body, input.ToString());

	protected sealed class InvalidOperatorHere : ParsingFailed
	{
		public InvalidOperatorHere(Body body, string message) : base(body, message) { }
	}

	protected sealed class UnknownExpression : ParsingFailed
	{
		public UnknownExpression(Body body, string error = "") : base(body, error) { }
	}

	protected sealed class CannotParseEmptyInput : ParsingFailed
	{
		public CannotParseEmptyInput(Body body) : base(body) { }
	}

	public sealed class ExpressionWithTypeAnyIsNotAllowed : ParsingFailed
	{
		public ExpressionWithTypeAnyIsNotAllowed(Body body, string message) : base(body, message) { }
	}

	protected sealed class NumbersCanNotBeInNestedCalls : ParsingFailed
	{
		public NumbersCanNotBeInNestedCalls(Body body, string text) : base(body, text) { }
	}

	public sealed class MemberOrMethodNotFound : ParsingFailed
	{
		public MemberOrMethodNotFound(Body body, Type? memberType, string memberName) : base(body,
			memberName, memberType) { }
	}

	protected sealed class UnknownExpressionForArgument : ParsingFailed
	{
		public UnknownExpressionForArgument(Body body, string message) : base(body, message) { }
	}

	protected sealed class ListTokensAreNotSeparatedByComma : ParsingFailed
	{
		public ListTokensAreNotSeparatedByComma(Body body) : base(body) { }
	}

	private sealed class InvalidSingleTokenExpression : ParsingFailed
	{
		public InvalidSingleTokenExpression(Body body, string message) : base(body, message) { } //ncrunch: no coverage
	}

	public sealed class InvalidArgumentItIsNotMethodOrListCall : ParsingFailed
	{
		public InvalidArgumentItIsNotMethodOrListCall(Body body, Expression variable,
			IEnumerable<Expression> arguments) : base(body, arguments.ToWordList(),
			variable.ReturnType) { }
	}
}