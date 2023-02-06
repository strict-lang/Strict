using System;
using static Strict.Language.Expressions.MethodExpressionParser;

namespace Strict.Language.Expressions;

public sealed class MemberCall : ConcreteExpression
{
	public MemberCall(Expression? instance, Member member) : base(member.Type)
	{
		Instance = instance;
		Member = member;
	}

	public Expression? Instance { get; }
	public Member Member { get; }

	public static Expression? TryParse(Body body, Type type, Expression? instance,
		ReadOnlySpan<char> partToParse)
	{
		foreach (var member in type.Members)
			if (partToParse.Equals(member.Name, StringComparison.Ordinal))
				return instance == null && body.IsFakeBodyForMemberInitialization
					? throw new CannotAccessMemberBeforeTypeIsParsed(body, partToParse.ToString(), type)
					: new MemberCall(instance, member);
#if LOG_DETAILS
		Logger.Info(nameof(TryParse) + " found no member in " + body.Method);
#endif
		return null;
	}

	public override string ToString() =>
		Instance != null
			? $"{Instance}.{Member.Name}"
			: Member.Name;
}