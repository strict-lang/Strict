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
		Assignment.TryParse(body, line) ?? If.TryParse(body, line) ??
		For.TryParse(body, line.Trim()) ?? Return.TryParse(body, line) ??
		Mutable.TryParse(body, line) ?? ParseExpression(body, line);

	public override Expression ParseExpression(Body body, ReadOnlySpan<char> input)
	{
		CheckIfEmptyOrAny(body, input);
		return input.Length < 3 || !input.Contains(' ') && !input.Contains(',') ||
			input.StartsWith(Base.Mutable)
				? TryParseCommon(body, input)
				: TryParse(body, input) ?? TryParseMethodOrMember(body, input);
	}

	private static void CheckIfEmptyOrAny(Body body, ReadOnlySpan<char> input)
	{
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(body);
		if (IsExpressionTypeAny(input))
			throw new ExpressionWithTypeAnyIsNotAllowed(body, input.ToString());
	}

	private Expression TryParseMethodOrMember(Body body, ReadOnlySpan<char> input)
	{
		var postfix = new ShuntingYard(input.ToString());
		if (postfix.Output.Count == 1)
			return TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ??
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

	private Expression TryParseCommon(Body body, ReadOnlySpan<char> input) =>
		Boolean.TryParse(body, input) ?? Text.TryParse(body, input) ??
		List.TryParseWithSingleElement(body, input) ?? Number.TryParse(body, input) ??
		Mutable.TryParse(body, input) ??
		TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ?? (input.IsOperator()
			? throw new InvalidOperatorHere(body, input.ToString())
			: input.IsWord()
				? throw new IdentifierNotFound(body, input.ToString())
				: throw new UnknownExpression(body, input.ToString()));

	private Expression? TryParse(Body body, ReadOnlySpan<char> input) =>
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

	private static bool IsExpressionTypeAny(ReadOnlySpan<char> input) =>
		input.Equals(Base.Any, StringComparison.Ordinal) || input.StartsWith(Base.Any + "(");

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
	/// By far the most common usecase, we call something from another instance, use some binary
	/// operator (like is, to, +, etc.) or execute some method. For more arguments more complex
	/// parsing has to be done and we have to invoke ShuntingYard for the argument list.
	/// </summary>
	public Expression? TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(Body body,
		ReadOnlySpan<char> input)
	{
		var argumentsStart = input.IndexOf('(');
		var argumentsEnd = input.FindMatchingBracketIndex(argumentsStart);
		ChangeArgumentStartEndIfNestedMethodCall(input, ref argumentsStart, ref argumentsEnd);
		if (argumentsStart > 0 && argumentsEnd > 0)
		{
			var call = ParseInContext(body.Method.Type, body, input[..argumentsStart],
					ParseListArguments(body, input[(argumentsStart + 1)..argumentsEnd])) ??
				TryParseListElementByIndex(body, input, input[(argumentsStart + 1)..argumentsEnd]);
			return argumentsEnd < input.Length - 1
				// ReSharper disable once TailRecursiveCall
				? TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input[(argumentsEnd + 2)..])
				: call;
		}
		return ParseInContext(body.Method.Type, body, input, Array.Empty<Expression>());
	}

	//https://deltaengine.fogbugz.com/f/cases/26205
	private static Expression? TryParseListElementByIndex(Body body, ReadOnlySpan<char> input,
		ReadOnlySpan<char> indexArgument)
	{
		var member = body.Method.ParseExpression(body, input[..input.IndexOf('(')]);
		if (member.ReturnType.IsList)
			return new ListCall(member, body.Method.ParseExpression(body, indexArgument));
		return (ListCall?)null;
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

	// ReSharper disable once TooManyArguments
	// ReSharper disable once MethodTooLong
	// ReSharper disable once CyclomaticComplexity
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
						List.TryParseWithMultipleOrNestedElements(body, input[members.Current]);
					if (current != null)
					{
						context = current.ReturnType;
						continue;
					}
				}
				var expression = input[members.Current].Contains('(')
					? TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input[members.Current])
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
					var referenceType = current?.ReturnType ?? body.Method.Type;
					throw new MemberOrMethodNotFound(body, null, input[members.Current].ToString() +
						($" in {referenceType}") + (current?.ReturnType != null
							? ParsingFailed.GetClickableStacktraceLine(current.ReturnType, 0, string.Empty)
							: string.Empty));
				}
				current = expression;
				context = current.ReturnType;
			}
			return TryListCall(body, current, arguments);
		}
		return TryListCall(body, TryMemberOrMethodCall(context, null, body, input, arguments),
			arguments);
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
		var variableValue = body.FindVariableValue(input);
		if (variableValue != null)
			return new VariableCall(input.ToString(), variableValue);
		if (input.Equals(Base.Value, StringComparison.Ordinal))
		{
			var valueInstance = Instance.Parse(body.Method);
			body.AddVariable(Base.Value, valueInstance);
			return valueInstance;
		}
		foreach (var parameter in body.Method.Parameters)
			if (input.Equals(parameter.Name, StringComparison.Ordinal))
				return new ParameterCall(parameter);
		var memberCall = TryFindMemberCall(type, instance, input);
		if (memberCall != null)
			return memberCall;
#if LOG_DETAILS
		Logger.Info(nameof(TryMemberOrMethodCall) + " found no member in " + body.Method);
#endif
		var methodName = input.ToString();
		var method2 = type.FindMethod(methodName, arguments);
		if (method2 != null)
			return new MethodCall(method2, instance, arguments);
		if (instance == null)
		{
			var fromType = body.Method.FindType(methodName);
			if (fromType != null)
			{
				if (IsConstructorUsedWithSameArgumentType(arguments, fromType))
					throw new ConstructorForSameTypeArgumentIsNotAllowed(body);
				return new MethodCall(fromType.GetMethod(Method.From, arguments), new From(fromType),
					arguments);
			}
		}
#if LOG_DETAILS
		Logger.Info("ParseNested found no local method in " + body.Method.Type + ": " + methodName);
#endif
		return null;
	}

	private static Expression? TryListCall(Body body, Expression? variable,
		IReadOnlyList<Expression> arguments) =>
		variable is null or MethodCall
			? variable
			: arguments.Count > 0
				? variable.ReturnType.IsList
					? new ListCall(variable, arguments[0])
					: throw new InvalidArgumentItIsNotMethodOrListCall(body, variable, arguments)
				: variable;

	private static bool
		IsConstructorUsedWithSameArgumentType(IReadOnlyList<Expression> arguments, Type fromType) =>
		arguments.Count == 1 && (fromType == arguments[0].ReturnType ||
			arguments[0].ReturnType is GenericType genericType && fromType == genericType.Generic);

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

	public class ConstructorForSameTypeArgumentIsNotAllowed : ParsingFailed
	{
		public ConstructorForSameTypeArgumentIsNotAllowed(Body body) : base(body) { }
	}

	/// <summary>
	/// Figures out if there are any bracket groups or if there is binary expression action going on.
	/// Could also contain strings, we don't know. Most of the time it will just be a bunch of values.
	/// <see cref="ShuntingYard" /> will only parse till the next comma, has to call this till the end.
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

	public sealed class ExpressionWithTypeAnyIsNotAllowed : ParsingFailed
	{
		public ExpressionWithTypeAnyIsNotAllowed(Body body, string message) : base(body, message) { }
	}

	public sealed class NumbersCanNotBeInNestedCalls : ParsingFailed
	{
		public NumbersCanNotBeInNestedCalls(Body body, string text) : base(body, text) { }
	}

	public sealed class MemberOrMethodNotFound : ParsingFailed
	{
		public MemberOrMethodNotFound(Body body, Type? memberType, string memberName) : base(body,
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
		public InvalidSingleTokenExpression(Body body, string message) : base(body, message) { }
	}

	// ReSharper disable once HollowTypeName
	private sealed class InvalidArgumentItIsNotMethodOrListCall : ParsingFailed
	{
		public InvalidArgumentItIsNotMethodOrListCall(Body body, Expression variable,
			IEnumerable<Expression> arguments) : base(body, arguments.ToWordList(),
			variable.ReturnType) { }
	}

}