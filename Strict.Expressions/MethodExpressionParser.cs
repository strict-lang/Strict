using System.Diagnostics;
using System.Runtime.CompilerServices;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	/// <summary>
	/// Slightly slower version that checks high-level expressions that can only occur at the line
	/// level like mutable, if, for (those increase methodLineNumber as well) and return.
	/// Every other expression can be nested and can appear anywhere.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public override Expression ParseLineExpression(Body body, ReadOnlySpan<char> line) =>
		Declaration.TryParse(body, line) ?? If.TryParse(body, line) ??
		For.TryParse(body, line.Trim()) ?? Return.TryParse(body, line) ??
		MutableReassignment.TryParse(body, line) ?? ParseExpression(body, line);

	public override Expression ParseExpression(Body body, ReadOnlySpan<char> input,
		bool makeMutable = false)
	{
		CheckIfEmptyOrAny(body, input);
		return input.Length < 3 || !input.Contains(' ') && !input.Contains(',')
			? TryParseCommon(body, input, makeMutable)
			: TryParseErrorOrTextOrListOrConditionalExpression(body, input, makeMutable) ??
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

	private Expression TryParseCommon(Body body, ReadOnlySpan<char> input, bool makeMutable) =>
		Boolean.TryParse(body, input) ?? Text.TryParse(body, input) ??
		List.TryParseWithSingleElement(body, input, makeMutable) ?? Number.TryParse(body, input) ??
		TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ?? (input.IsOperator()
			? throw new InvalidOperatorHere(body, input.ToString())
			: input.IsWord()
				? throw new Body.IdentifierNotFound(body, input.ToString())
				: throw new UnknownExpression(body, input.ToString()));

	private static Expression? TryParseErrorOrTextOrListOrConditionalExpression(Body body,
		ReadOnlySpan<char> input, bool makeMutable) =>
		input[0] == '"' && input[^1] == '"' && MemoryExtensions.Count(input, '"') == 2
			? new Text(body.Method, input.Slice(1, input.Length - 2).ToString())
			: input[0] == '(' && input[^1] == ')' && input.Contains(',') &&
			MemoryExtensions.Count(input, '(') == 1
				? new List(body, body.Method.ParseListArguments(body, input[1..^1]), makeMutable)
				: If.CanTryParseConditional(body, input)
					? If.ParseConditional(body, input)
					: null;

	private Expression TryParseMethodOrMember(Body body, ReadOnlySpan<char> input)
	{
		var inputText = input.ToString();
		var postfix = new ShuntingYard(inputText);
		if (postfix.Output.Count == 1)
			return Dictionary.TryParse(body, input) ??
				TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ??
				ParseTextWithSpacesOrListWithMultipleOrNestedElements(body, input[postfix.Output.Pop()]);
		if (postfix.Output.Count == 2)
			return ParseMethodCallWithArguments(body, input, postfix);
		var binary = Binary.Parse(body, input, postfix.Output);
		if (postfix.Output.Count == 0)
			return
#if DEBUG
				inputText != binary.ToString()
					? throw new GeneratedBinaryExpressionDoesNotMatchInputExactly(body, binary, inputText)
					:
#endif
					binary;
		return ParseInContext(body, input[postfix.Output.Peek()], [binary]) ??
			throw new UnknownExpression(body,
				input[postfix.Output.Peek()].ToString() + " in " + inputText);
	}

	private sealed class GeneratedBinaryExpressionDoesNotMatchInputExactly(Body body,
		Expression binary, string inputText) : ParsingFailed(body, binary + ", inputText=" + inputText);

	private Expression ParseMethodCallWithArguments(Body body, ReadOnlySpan<char> input,
		ShuntingYard postfix)
	{
		var argumentsRange = postfix.Output.Pop();
		var methodRange = postfix.Output.Pop();
		return input[argumentsRange.Start.Value] == '('
			? ParseInContext(body, input[methodRange], ParseListArguments(body,
				input[(argumentsRange.Start.Value + 1)..(argumentsRange.End.Value - 1)])) ??
			throw new MemberOrMethodNotFound(body, body.Method.Type, input[methodRange].ToString())
			: input[argumentsRange.Start.Value] == '.'
				? ParseInContext(body, input, []) ??
				throw new InvalidOperatorHere(body, input[methodRange].ToString())
				: input[argumentsRange].Equals(UnaryOperator.Not, StringComparison.Ordinal)
					? Not.Parse(body, input, methodRange)
					: input[0].IsSingleCharacterOperator() && IsContextInForExpression(body)
						? ParseExpression(body, input[2..])
						: throw new InvalidOperatorHere(body, input[methodRange].ToString());
	}

	private static bool IsContextInForExpression(Body body) =>
		body.Parent != null && body.Parent.GetLine(body.LineRange.Start.Value - 1).TrimStart().
			StartsWith(Keyword.For, StringComparison.Ordinal);

	/// <summary>
	/// By far the most common use-case, we call something from another instance, use some binary
	/// operator (like "is, to, +", etc.) or execute some method. For more arguments more complex
	/// parsing has to be done, and we have to invoke ShuntingYard for the argument list.
	/// </summary>
	private Expression? TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(Body body,
		ReadOnlySpan<char> input)
	{
		var argumentsStart = input.IndexOf('(');
		var argumentsEnd = input.FindMatchingBracketIndex(argumentsStart);
		ChangeArgumentStartEndIfNestedMethodCall(input, ref argumentsStart, ref argumentsEnd);
		if (argumentsStart <= 0 || argumentsEnd <= 0 || argumentsEnd < input.Length - 1)
			return ParseInContext(body, input, []);
		return ParseInContext(body, input[..argumentsStart],
			ParseListArguments(body, input[(argumentsStart + 1)..argumentsEnd]));
	}

	private static void ChangeArgumentStartEndIfNestedMethodCall(ReadOnlySpan<char> input,
		ref int argumentsStart, ref int argumentsEnd)
	{
		if (!IsNestedMethodCallWithParentMethodParameter(input, argumentsStart, argumentsEnd))
			return;
		argumentsStart = input.LastIndexOf('(');
		argumentsEnd = input.FindMatchingBracketIndex(argumentsStart);
	}

	private static bool IsNestedMethodCallWithParentMethodParameter(ReadOnlySpan<char> input,
		int argumentsStart, int argumentsEnd)
	{
		var innerArgumentStart = input.LastIndexOf('(');
		return argumentsStart != innerArgumentStart && argumentsEnd < innerArgumentStart &&
			input.IndexOf('.') < innerArgumentStart;
	}

	private Expression? ParseInContext(Body body, ReadOnlySpan<char> input,
		IReadOnlyList<Expression> arguments) =>
		input.Contains('.')
			? ParseNestedExpressionInContext(body, input, arguments)
			: ListCall.TryParse(body,
				TryVariableOrValueOrParameterOrMemberOrMethodCall(body.Method.Type, null, body, input,
					arguments), arguments);

	private Expression? ParseNestedExpressionInContext(Body body,
		ReadOnlySpan<char> input, IReadOnlyList<Expression> arguments)
	{
		var members = new RangeEnumerator(input, '.', 0);
		var context = body.Method.Type;
		Expression? current = null;
		while (members.MoveNext())
		{
			if (current is null)
			{
				current = Text.TryParse(body, input[members.Current]) ??
					List.TryParseWithMultipleOrNestedElements(body, input[members.Current], false);
				if (current is not null)
				{
					context = current.ReturnType;
					continue;
				}
			}
			var expression = input[members.Current].Contains('(')
				? TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input[members.Current])
				: TryVariableOrValueOrParameterOrMemberOrMethodCall(context, current, body,
					input[members.Current],
					// arguments are only needed for the last part
					members.IsAtEnd
						? arguments
						: []);
			current = expression ??
				throw CheckErrorTypeAndThrowException(body, input, members, current);
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

	// ReSharper disable once TooManyArguments
	private Expression? TryVariableOrValueOrParameterOrMemberOrMethodCall(Context context,
		Expression? instance, Body body, ReadOnlySpan<char> input, IReadOnlyList<Expression> arguments)
	{
		var inputAsString = input.ToString();
		var type = context as Type ?? body.Method.Type;
		if (!input.IsWord() && !input.Contains(' ') && !input.Contains('('))
			return inputAsString.IsWordOrWordWithNumberAtEnd(out _)
				? MethodCall.TryParseFromOrEnum(body, arguments, inputAsString)
				: null;
		var call = VariableCall.TryParse(body, input) ??
			(input.Equals(Base.ValueLowercase, StringComparison.Ordinal)
				? Instance.Parse(body, body.Method)
				: ParameterCall.TryParse(body, input));
		if (call != null)
			return call;
		if (inputAsString.IsKeyword())
			throw new KeywordNotAllowedAsMemberOrMethod(body, inputAsString, type);
		var parse = MemberCall.TryParse(body, type, instance, input) ??
			MethodCall.TryParse(instance, body, arguments, type, input.ToString());
		if (parse == null && instance is null)
			parse = MethodCall.TryParseFromOrEnum(body, arguments, inputAsString);
		if (parse != null)
			return parse;
		if (arguments.Count > 0 && input.EndsWith(')'))
			return TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input);
		return null;
	}

	public sealed class CannotAccessMemberBeforeTypeIsParsed(Body body, string input, Type type)
		: ParsingFailed(body, input, type);

	public sealed class KeywordNotAllowedAsMemberOrMethod(Body body, string input, Type type)
		: ParsingFailed(body, input, type);

	/// <summary>
	/// Figures out if there are any bracket groups or if there is a binary expression going on.
	/// Could also contain strings, we don't know. Most of the time it will just be some values.
	/// <see cref="ShuntingYard" /> will parse till the next comma, has to call this till the end.
	/// </summary>
	public override List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> innerSpan)
	{
		if (innerSpan.Contains('(') || innerSpan.Contains('"') && innerSpan.Contains(' '))
			return If.CanTryParseConditional(body, innerSpan)
				? [If.ParseConditional(body, innerSpan)]
				: new ExpressionListParser(this, innerSpan.ToString()).GetAll(body);
		return innerSpan.Length == 0
			? throw new List.EmptyListNotAllowed(body)
			: ParseAllElementsFast(body, innerSpan, new RangeEnumerator(innerSpan, ',', 0));
	}

	public override bool IsVariableMutated(Body body, string variableName)
	{
		foreach (var expression in body.Expressions)
		{
			if (IsMutationOfVariable(expression, variableName) ||
				expression is Body childBody && IsVariableMutated(childBody, variableName) ||
				expression is If ifExpression && CheckForVariableMutationInIf(variableName, ifExpression))
				return true;
			if (expression is For forExpression &&
				(IsMutationOfVariable(forExpression.Body, variableName) ||
					forExpression.Iterator is Value forValue && forValue.Data.ToString() == variableName ||
					forExpression.Iterator is Binary { Instance: VariableCall forVariableCall } &&
					forVariableCall.Variable.Name == variableName || forExpression.Body is Body forBody &&
					IsVariableMutated(forBody, variableName)))
				return true;
			if (expression is MethodCall { Instance: VariableCall variableCall, IsMutable: true } &&
				variableCall.Variable.Name == variableName)
				return true;
		}
		return false;
	}

	private static bool IsMutationOfVariable(Expression expression, string variableName) =>
		expression is MutableReassignment reassignment && (reassignment.Name == variableName ||
			reassignment.Target is ListCall { List: VariableCall listCall } &&
			listCall.Variable.Name == variableName);

	private bool CheckForVariableMutationInIf(string variableName, If ifExpression)
	{
		if (IsMutationOfVariable(ifExpression.Then, variableName) ||
			ifExpression.Then is Body thenBody && IsVariableMutated(thenBody, variableName) ||
			ifExpression.Then is If ifBody && CheckForVariableMutationInIf(variableName, ifBody))
			return true;
		return ifExpression.OptionalElse != null &&
			(IsMutationOfVariable(ifExpression.OptionalElse, variableName) ||
				ifExpression.OptionalElse is Body elseBody && IsVariableMutated(elseBody, variableName) ||
				ifExpression.OptionalElse is If elseIfBody &&
				CheckForVariableMutationInIf(variableName, elseIfBody));
	}

	/// <summary>
	/// Similar to TryParseExpression, but we know there are commas separating expressions
	/// </summary>
	public class ExpressionListParser(MethodExpressionParser parser, string inner)
	{
		private readonly ShuntingYard postfix = new(inner);

		/// <summary>
		/// The postfix data comes in upside down, so use another stack to restore order
		/// </summary>
		public List<Expression> GetAll(Body body)
		{
			var expressions = new Stack<Expression>();
			if (postfix.Output.Count == 1)
				expressions.Push(parser.ParseTextWithSpacesOrListWithMultipleOrNestedElements(body,
					inner[postfix.Output.Pop()]));
			else if (postfix.Output.Count == 2)
				expressions.Push(parser.ParseMethodCallWithArguments(body, inner.AsSpan(), postfix)); //ncrunch: no coverage
			else
				ParseBinaryOrNormalExpressionsIntoList(body, expressions);
			return [..expressions];
		}

		private void ParseBinaryOrNormalExpressionsIntoList(Body body, Stack<Expression> expressions)
		{
			do
			{
				var span = inner[postfix.Output.Peek()];
				try
				{
					// Is this a binary expression we have to put into the list (tokenized and postfixed)?
					expressions.Push(span.Length == 1 && span[0].IsSingleCharacterOperator() ||
						span.IsMultiCharacterOperator()
							? Binary.Parse(body, inner.AsSpan(), postfix.Output)
							: body.Method.ParseExpression(body, inner[postfix.Output.Pop()]));
				}
				catch (UnknownExpression ex)
				{
					throw new UnknownExpressionForArgument(body,
						span + " is invalid for argument " + expressions.Count + " " + ex.Message);
				}
				if (postfix.Output.Count > 0 && inner[postfix.Output.Pop().Start.Value] != ',')
					throw new ListTokensAreNotSeparatedByComma(body);
			} while (postfix.Output.Count > 0);
		}
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
		return expressions;
	}

	private Expression
		ParseTextWithSpacesOrListWithMultipleOrNestedElements(Body body, ReadOnlySpan<char> input) =>
		Text.TryParse(body, input) ?? List.TryParseWithMultipleOrNestedElements(body, input, false) ??
		TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ??
		throw new InvalidSingleTokenExpression(body, input.ToString());

	protected sealed class InvalidOperatorHere(Body body, string message)
		: ParsingFailed(body, message);

	protected sealed class UnknownExpression(Body body, string error = "")
		: ParsingFailed(body, error);

	protected sealed class CannotParseEmptyInput(Body body) : ParsingFailed(body);

	public sealed class ExpressionWithTypeAnyIsNotAllowed(Body body, string message)
		: ParsingFailed(body, message);

	protected sealed class NumbersCanNotBeInNestedCalls(Body body, string text)
		: ParsingFailed(body, text);

	public sealed class MemberOrMethodNotFound(Body body, Type? memberType, string memberName)
		: ParsingFailed(body, memberName, memberType);

	protected sealed class UnknownExpressionForArgument(Body body, string message)
		: ParsingFailed(body, message);

	protected sealed class ListTokensAreNotSeparatedByComma(Body body) : ParsingFailed(body);

	private sealed class InvalidSingleTokenExpression(Body body, string message)
		: ParsingFailed(body, message); //ncrunch: no coverage

	public sealed class InvalidArgumentItIsNotMethodOrListCall(Body body,
		Expression variable, IEnumerable<Expression> arguments)
		: ParsingFailed(body, arguments.ToWordList(), variable.ReturnType);
}