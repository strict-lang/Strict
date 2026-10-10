using System.Runtime.CompilerServices;
using System.Text;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public partial class MethodExpressionParser : ExpressionParser
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
		return input.Length < 3 || (!input.Contains(' ') && !input.Contains(','))
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
		input.Equals(Type.Any, StringComparison.Ordinal) || input.StartsWith(Type.Any + "(");

	private Expression TryParseCommon(Body body, ReadOnlySpan<char> input, bool makeMutable) =>
		Boolean.TryParse(body, input) ?? Text.TryParse(body, input) ??
		List.TryParseWithSingleElement(body, input, makeMutable) ?? Number.TryParse(body, input) ??
		TryParseForBodyValueMethodCall(body, input) ??
		TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ??
		TryParseConstraintBodyMethodCall(body, input) ?? (input.IsOperator()
			? throw new InvalidOperatorHere(body, input.ToString())
			: input.IsWord()
				? throw new Body.IdentifierNotFound(body, input.ToString())
				: throw new UnknownExpression(body, DescribeUnknown(body, input.ToString())));

	private static Expression? TryParseForBodyValueMethodCall(Body body, ReadOnlySpan<char> input)
	{
		if (!IsContextInForExpression(body) || !input.IsWord())
			return null;
		var valueVar = body.FindVariable(Type.ValueLowercase.AsSpan());
		if (valueVar == null)
			return null; //ncrunch: no coverage
		var inputName = input.ToString();
		var valueType = valueVar.Type;
		var valueCall = new VariableCall(valueVar, body.CurrentFileLineNumber);
		if (input.Equals(Type.ValueLowercase, StringComparison.Ordinal))
			return valueCall;
		var localOrParameter = TryParseLocalVariableOrParameter(body, input);
		if (localOrParameter != null)
			return localOrParameter;
		Method? method;
		try
		{
			method = valueType.FindMethod(inputName, []);
		}
		catch (Type.GenericTypesCannotBeUsedDirectlyUseImplementation)
		{
			return null;
		}
		if (method is { IsTrait: false })
			return new MethodCall(method, valueCall, [], null,
				body.CurrentFileLineNumber); //ncrunch: no coverage
		var member = FindMember(valueType, inputName);
		return member != null
			? new MemberCall(valueCall, member, body.CurrentFileLineNumber)
			: null;
	}

	private static Expression? TryParseConstraintBodyMethodCall(Body body, ReadOnlySpan<char> input)
	{
		if (body.Method.Name != Member.ConstraintsBody || !input.IsWord())
			return null;
		var constraintType = body.Method.Type;
		var inputName = input.ToString();
		var method = constraintType.FindMethod(inputName, []);
		return method != null
			? new MethodCall(method, null, [], null, body.CurrentFileLineNumber)
			: null;
	}

	private static Member? FindMember(Type valueType, string inputName)
	{
		for (var index = 0; index < valueType.Members.Count; index++)
			//ncrunch: no coverage start
			if (valueType.Members[index].Name.Equals(inputName, StringComparison.Ordinal))
				return valueType.Members[index];
		//ncrunch: no coverage end
		return null;
	}

	private static Expression? TryParseErrorOrTextOrListOrConditionalExpression(Body body,
		ReadOnlySpan<char> input, bool makeMutable) =>
		input[0] == '"' && input[^1] == '"' && MemoryExtensions.Count(input, '"') == 2
			? Text.TryParse(body, input)
			: input[0] == '(' && input[^1] == ')' && IsSingleBracketedList(input)
				? new List(body, body.Method.ParseListArguments(body, input[1..^1]), makeMutable)
				: If.CanTryParseConditional(body, input)
					? If.ParseConditional(body, input)
					: null;

	/// <summary>
	/// A comma directly inside the outer bracket makes a list: (1, (a then 2 else 3)) is a list,
	/// (a then (1, 2) else (3, 4)) is a conditional.
	/// </summary>
	private static bool IsSingleBracketedList(ReadOnlySpan<char> input)
	{
		var nestedBracketDepth = 0;
		var topLevelOpenBracketCount = 0;
		var hasTopLevelComma = false;
		var isInsideText = false;
		for (var index = 0; index < input.Length; index++)
		{
			isInsideText = TextLiteral.Advance(input, ref index, isInsideText);
			if (isInsideText)
				continue;
			if (input[index] == '(')
			{
				nestedBracketDepth++;
				if (nestedBracketDepth == 1)
					topLevelOpenBracketCount++;
			}
			else if (input[index] == ')')
				nestedBracketDepth--;
			else if (input[index] == ',' && nestedBracketDepth == 1)
				hasTopLevelComma = true;
		}
		return topLevelOpenBracketCount == 1 && nestedBracketDepth == 0 && hasTopLevelComma;
	}

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
#if DEBUG
			return AreEquivalentExpressionTexts(body, inputText, binary.ToString())
				? binary
				: throw new GeneratedBinaryExpressionDoesNotMatchInputExactly(body, binary, inputText);
#else
			return binary;
#endif
		return ParseInContext(body, input[postfix.Output.Peek()], [binary]) ??
			throw new UnknownExpression(body,
				input[postfix.Output.Peek()].ToString() + " in " + inputText);
	}

#if DEBUG
	private static bool
		AreEquivalentExpressionTexts(Body body, string inputText, string generatedText) =>
		inputText == generatedText || NormalizeExpressionText(body, inputText) ==
		NormalizeExpressionText(body, generatedText);

#endif
	private Expression ParseMethodCallWithArguments(Body body, ReadOnlySpan<char> input,
		ShuntingYard postfix)
	{
		var argumentsRange = postfix.Output.Pop();
		var methodRange = postfix.Output.Pop();
		return input[argumentsRange.Start.Value] == '('
			? ParseInContext(body, input[methodRange],
				ParseListArguments(body,
					input[(argumentsRange.Start.Value + 1)..(argumentsRange.End.Value - 1)])) ??
			throw new MemberOrMethodNotFound(body, body.Method.Type, input[methodRange].ToString())
			: input[argumentsRange.Start.Value] == '.'
				? ParseInContext(body, input, []) ??
				throw new InvalidOperatorHere(body, input[methodRange].ToString())
				: input[argumentsRange].Equals(UnaryOperator.Not, StringComparison.Ordinal)
					? Not.Parse(body, input, methodRange)
					: input[0].IsSingleCharacterOperator() && IsContextInForExpression(body)
						? ParseExpression(body, input[2..])
						: input[argumentsRange].IsMultiCharacterOperator() && IsContextInForExpression(body)
							? ParseExpression(body, "value " + input.ToString())
							: throw new InvalidOperatorHere(body, input[methodRange].ToString());
	}

	private static bool IsContextInForExpression(Body body)
	{
		var current = body;
		while (current.Parent != null)
		{
			if (current.Parent.GetLine(current.LineRange.Start.Value - 1).TrimStart().
				StartsWith(Keyword.For, StringComparison.Ordinal))
				return true;
			current = current.Parent;
		}
		return false;
	}

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
		var argumentsText = input[(argumentsStart + 1)..argumentsEnd];
		// If our arguments are types, we might have ended up from a generic constructor like
		// Dictionary(Number, Number) here, construct the type and return that method!
		var resolvedType = body.Method.FindType(input[..argumentsStart].ToString());
		if (resolvedType is { IsGeneric: true })
		{
			var typeArgNames = argumentsText.ToString().Split(", ");
			if (typeArgNames.Length == resolvedType.GetGenericTypeArguments().Count &&
				typeArgNames.All(n => body.Method.FindType(n) != null))
			{
				var fromType = resolvedType.GetGenericImplementation(
					typeArgNames.Select(t => body.Method.Type.GetType(t)).ToArray());
				return MethodCall.CreateFromMethodCall(body, fromType, []);
			}
		}
		return ParseInContext(body, input[..argumentsStart], ParseListArguments(body, argumentsText));
	}

	private Expression?
		TryParseConstraintRoot(Body body, Type context, ReadOnlySpan<char> inputText) =>
		body.Method.Name == Member.ConstraintsBody
			? TryVariableOrValueOrParameterOrMemberOrMethodCall(context, null, body, inputText, [])
			: null;

	// ReSharper disable once TooManyArguments
	private Expression? TryVariableOrValueOrParameterOrMemberOrMethodCall(Context context,
		Expression? instance, Body body, ReadOnlySpan<char> input, IReadOnlyList<Expression> arguments)
	{
		var type = context as Type ?? body.Method.Type;
		return !input.IsWord() && !input.Contains(' ') && !input.Contains('(')
			? TryParseStandaloneToken(body, arguments, input)
			: TryParseOuterVariable(body, input, instance) ??
			// Instance members must win over same-named parameters (FromNumber(3).number).
			(instance is null
				? TryParseLocalVariableOrParameter(body, input)
				: null) ?? TryParseDictionaryElementsAlias(body, type, instance, input) ??
			TryParseGenericTypeEnum(body, type, instance, arguments, input) ??
			TryParseMemberOrMethodCall(instance, body, input, arguments, type);
	}

	private Expression? TryParseMemberOrMethodCall(Expression? instance, Body body,
		ReadOnlySpan<char> input, IReadOnlyList<Expression> arguments, Type type)
	{
		if (input.IsKeyword())
			throw new KeywordNotAllowedAsMemberOrMethod(body, input.ToString(), type);
		if (instance is null && type != body.Method.Type &&
			input.Equals(Method.From, StringComparison.Ordinal))
			throw new DirectFromConstructorCallIsForbidden(body, type);
		var parse = (instance is null && input.Equals(body.Method.Type.Name, StringComparison.Ordinal)
				? MethodCall.TryParseFromOrEnum(body, arguments, input.ToString())
				: null) ?? MemberCall.TryParse(body, type, instance, input) ??
			MethodCall.TryParse(instance, body, arguments, type, input.ToString());
		if (parse == null && instance is null)
			parse = MethodCall.TryParseFromOrEnum(body, arguments, input.ToString()) ??
				TryParseForBodyValueMethodCallWithArguments(body, input, arguments);
		if (parse != null)
			return parse;
		if (arguments.Count > 0 && input.EndsWith(')'))
			return TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input);
		return null;
	}

	private static Expression? TryParseForBodyValueMethodCallWithArguments(Body body,
		ReadOnlySpan<char> input, IReadOnlyList<Expression> arguments)
	{
		if (!IsContextInForExpression(body))
			return null;
		var valueVar = body.FindVariable(Type.ValueLowercase.AsSpan());
		if (valueVar == null)
			return null;
		var valueCall = new VariableCall(valueVar, body.CurrentFileLineNumber);
		var method = valueVar.Type.FindMethod(input.ToString(), arguments);
		return method != null
			? new MethodCall(method, valueCall, arguments, null, body.CurrentFileLineNumber)
			: null;
	}

	private static Expression? TryParseStandaloneToken(Body body, IReadOnlyList<Expression> arguments,
		ReadOnlySpan<char> input) =>
		input.IsWordOrWordWithNumberAtEnd(out _)
			? MethodCall.TryParseFromOrEnum(body, arguments, input.ToString())
			: null;

	private static Expression? TryParseOuterVariable(Body body, ReadOnlySpan<char> input,
		Expression? instance)
	{
		if (input.Equals(Type.OuterLowercase, StringComparison.Ordinal))
		{
			var methodType = body.Method.Type;
			var outerInstanceType = methodType.IsGeneric && methodType is not GenericTypeImplementation
				? body.Method.GetType(Type.Any)
				: methodType;
			return new VariableCall(
				new Variable(Type.OuterLowercase, false,
					new Instance(outerInstanceType, body.CurrentFileLineNumber), body),
				body.CurrentFileLineNumber);
		}
		if (instance is not VariableCall { Variable.Name: Type.OuterLowercase })
			return null;
		var call = VariableCall.TryParse(body.Parent!, input) ??
			(input.Equals(Type.ValueLowercase, StringComparison.Ordinal)
				? Instance.Parse(body.Parent!, body.Method)
				: ParameterCall.TryParse(body, input));
		return call == null
			? null
			: new MemberCall(instance, new Member(body.ReturnType, input.ToString(), call.ReturnType),
				body.CurrentFileLineNumber);
	}

	private static Expression?
		TryParseLocalVariableOrParameter(Body body, ReadOnlySpan<char> input) =>
		VariableCall.TryParse(body, input) ??
		(input.Equals(Type.ValueLowercase, StringComparison.Ordinal)
			? Instance.Parse(body, body.Method)
			: ParameterCall.TryParse(body, input));

	private static Expression? TryParseDictionaryElementsAlias(Body body, Type type,
		Expression? instance, ReadOnlySpan<char> input)
	{
		if (!input.Equals(Type.ElementsLowercase, StringComparison.Ordinal) ||
			type is not GenericTypeImplementation { Generic.Name: Type.Dictionary })
			return null;
		var listMember = type.Members.FirstOrDefault(member =>
			member.Type.Name.StartsWith(Type.List, StringComparison.Ordinal));
		if (listMember == null)
			return null; //ncrunch: no coverage
		var keyword = listMember.IsConstant
			? Keyword.Constant
			: listMember.IsMutable
				? Keyword.Mutable
				: Keyword.Has;
		var aliasMember = new Member(type, $"{Type.ElementsLowercase} {listMember.Type.Name}", null,
			listMember.LineNumber, keyword);
		return new MemberCall(instance, aliasMember, body.CurrentFileLineNumber);
	}

	/// <summary>
	/// If inside a generic type that cannot be used directly, we still might have a declaration or
	/// Enum usage. This won't use the outer generic type, but might be needed for tests (e.g. Error)
	/// </summary>
	private static Expression? TryParseGenericTypeEnum(Body body, Type type, Expression? instance,
		IReadOnlyList<Expression> arguments, ReadOnlySpan<char> input) =>
		instance is null && type.IsGeneric && input.IsWordOrWordWithNumberAtEnd(out _) &&
		(arguments.Count > 0 || (MemberCall.TryParse(body, type, null, input) == null &&
			!type.AvailableMethods.ContainsKey(input.ToString())))
			? MethodCall.TryParseFromOrEnum(body, arguments, input.ToString())
			: null;

	private Expression
		ParseTextWithSpacesOrListWithMultipleOrNestedElements(Body body, ReadOnlySpan<char> input) =>
		Text.TryParse(body, input) ?? List.TryParseWithMultipleOrNestedElements(body, input, false) ??
		TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, input) ??
		throw new InvalidSingleTokenExpression(body, input.ToString());
}
