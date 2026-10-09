using System.Runtime.CompilerServices;
using System.Text;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public partial class MethodExpressionParser
{
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

	public sealed class CannotAccessMemberBeforeTypeIsParsed(Body body, string input, Type type)
		: ParsingFailed(body, input, type);

	public sealed class DirectFromConstructorCallIsForbidden(Body body, Type type)
		: ParsingFailed(body, "Use " + type.Name + "(...) instead of " + type.Name + ".from(...)",
			type);

	public sealed class KeywordNotAllowedAsMemberOrMethod(Body body, string input, Type type)
		: ParsingFailed(body, input, type);

	protected sealed class InvalidOperatorHere(Body body, string message)
		: ParsingFailed(body, message);

	private static string DescribeUnknown(Body body, string input)
	{
		var open = input.IndexOf('(');
		if (open <= 0 || !input.EndsWith(')') || !char.IsUpper(input[0]))
			return input;
		var typeName = input[..open];
		return body.Method.FindType(typeName) != null
			? input
			: "Type \"" + typeName + "\" is not in this package. Add " + typeName + ".strict next to " +
			body.Method.Type.Name + ".strict or check the spelling.";
	}

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
		Expression variable,
		IReadOnlyList<Expression> arguments)
		: ParsingFailed(body, string.Join(", ", arguments), variable.ReturnType);
}
