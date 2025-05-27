using static Strict.Language.Expressions.MethodExpressionParser;

namespace Strict.Language.Expressions;

public sealed class MemberCall(Expression? instance, Member member)
	: ConcreteExpression(member.Type, member.IsMutable)
{
	public Expression? Instance { get; } = instance;
	public Member Member { get; } = member;

	// ReSharper disable once TooManyArguments
	public static Expression? TryParse(Body body, Type type, Expression? instance,
		ReadOnlySpan<char> partToParse)
	{
		foreach (var member in type.Members)
			if (partToParse.Equals(member.Name, StringComparison.Ordinal))
				return instance == null && body.IsFakeBodyForMemberInitialization
					? throw new CannotAccessMemberBeforeTypeIsParsed(body, partToParse.ToString(), type)
					: new MemberCall(instance, member);
		return null;
	}

	public override string ToString() =>
		Instance != null
			? $"{Instance}.{Member.Name}"
			: Member.Name;
}