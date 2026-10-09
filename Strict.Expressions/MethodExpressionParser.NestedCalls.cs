using System.Runtime.CompilerServices;
using System.Text;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public partial class MethodExpressionParser
{
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
		ContainsMemberSeparatorOutsideBrackets(input)
			? ParseNestedExpressionInContext(body, input, arguments)
			: ListCall.TryParse(body,
				TryVariableOrValueOrParameterOrMemberOrMethodCall(body.Method.Type, null, body, input,
					arguments), arguments);

	private static bool ContainsMemberSeparatorOutsideBrackets(ReadOnlySpan<char> input)
	{
		var bracketCount = 0;
		var inText = false;
		foreach (var current in input)
		{
			if (current == '"')
				inText = !inText;
			if (inText)
				continue;
			if (current == '(')
				bracketCount++;
			else if (current == ')')
				bracketCount--;
			else if (current == '.' && bracketCount == 0)
				return true;
		}
		return false;
	}

	//TODO: this method is way too long
	private Expression? ParseNestedExpressionInContext(Body body, ReadOnlySpan<char> input,
		IReadOnlyList<Expression> arguments)
	{
		var nestedInput = input;
		if (nestedInput.StartsWith(Type.ValueLowercase + ".", StringComparison.Ordinal) &&
			body.FindVariable(Type.ValueLowercase.AsSpan()) == null)
			Instance.Parse(body, body.Method);
		Expression? current = null;
		var context = body.Method.Type;
		var callArguments = arguments;
		if (TryParseLeadingNumberInstance(body, ref nestedInput, ref current, ref context))
			if (nestedInput.Length > 0 && nestedInput[0] == '.')
			{
				if (arguments.Count == 1 && arguments[0] is Binary)
				{
					current = arguments[0];
					context = current.ReturnType;
					callArguments = [];
					nestedInput = nestedInput[1..];
				}
				else
				{
					throw new InvalidOperatorHere(body, nestedInput.ToString());
				}
			}
		var members = new RangeEnumerator(nestedInput, '.', 0);
		while (members.MoveNext())
		{
			if (current is null)
			{
				var inputText = nestedInput[members.Current];
				if (inputText.Length >= 3 && inputText[0] == PhraseTokenizer.OpenBracket)
				{
					var postfix = new ShuntingYard(inputText.ToString());
					if (postfix.Output.Count >= 3)
						current = Binary.Parse(body, nestedInput, postfix.Output);
				}
				current ??= Text.TryParse(body, inputText) ??
					List.TryParseWithMultipleOrNestedElements(body, inputText, false) ??
					Dictionary.TryParse(body, inputText) ?? (inputText.Length > 0 &&
						(char.IsDigit(inputText[0]) || inputText[0] == '-')
							? Number.TryParse(body, inputText)
							: null);
				if (current is not null)
				{
					context = current.ReturnType;
					continue;
				}
				current = TryParseConstraintRoot(body, context, inputText);
				if (current is not null)
				{
					context = current.ReturnType;
					continue;
				}
				var foundType = body.Method.FindType(inputText.ToString());
				if (foundType != null && !members.IsAtEnd && members.Current.Start.Value == 0 &&
					body.Method.Type.FindMember(inputText.ToString()) == null &&
					!inputText.Equals(Type.ValueLowercase, StringComparison.Ordinal) &&
					!inputText.Equals(Type.OuterLowercase, StringComparison.Ordinal))
				{
					context = foundType;
					continue;
				}
			}
			if (current != null)
			{
				var part = nestedInput[members.Current];
				var partName = part.Contains('(')
					? part[..part.IndexOf('(')]
					: part;
				if (partName.IsOperator())
					throw new InvalidOperatorHere(body, partName.ToString());
			}
			var expression = nestedInput[members.Current].Contains('(')
				? current != null || context != body.Method.Type
					? ParseMethodCallOnContext(body, nestedInput[members.Current], context, current)
					: TryParseMemberOrZeroOrOneArgumentMethodOrNestedCall(body, nestedInput[members.Current])
				: TryVariableOrValueOrParameterOrMemberOrMethodCall(context, current, body,
					nestedInput[members.Current], members.IsAtEnd
						? callArguments
						: []);
			current = expression ??
				throw CheckErrorTypeAndThrowException(body, nestedInput, members, current);
			context = current.ReturnType;
		}
		return ListCall.TryParse(body, current, callArguments);
	}

	private static bool TryParseLeadingNumberInstance(Body body, ref ReadOnlySpan<char> nestedInput,
		ref Expression? current, ref Type context)
	{
		var numberStart = nestedInput.Length > 1 && nestedInput[0] == '-' &&
			char.IsDigit(nestedInput[1])
				? 1
				: 0;
		if (nestedInput.IsEmpty || nestedInput.Length <= numberStart ||
			!char.IsDigit(nestedInput[numberStart]))
			return false;
		var numberLength = numberStart + GetLeadingNumberLength(nestedInput[numberStart..]);
		if (numberLength <= numberStart || numberLength >= nestedInput.Length ||
			nestedInput[numberLength] != '.')
			return false;
		var leadingNumber = Number.TryParse(body, nestedInput[..numberLength]);
		if (leadingNumber == null)
			return false;
		current = leadingNumber;
		context = leadingNumber.ReturnType;
		nestedInput = nestedInput[(numberLength + 1)..];
		return true;
	}

	private static int GetLeadingNumberLength(ReadOnlySpan<char> input)
	{
		var index = 0;
		while (index < input.Length && char.IsDigit(input[index]))
			index++;
		if (index < input.Length && input[index] == '.' && index + 1 < input.Length &&
			char.IsDigit(input[index + 1]))
		{
			index++;
			while (index < input.Length && char.IsDigit(input[index]))
				index++;
		}
		return index;
	}

	private Expression? ParseMethodCallOnContext(Body body, ReadOnlySpan<char> input, Context context,
		Expression? current)
	{
		var argStart = input.IndexOf('(');
		var argEnd = input.FindMatchingBracketIndex(argStart);
		var args = argEnd > argStart + 1
			? ParseListArguments(body, input[(argStart + 1)..argEnd])
			: (IReadOnlyList<Expression>)[];
		return ListCall.TryParse(body, TryVariableOrValueOrParameterOrMemberOrMethodCall(context,
			current, body, input[..argStart], args), args);
	}
}
