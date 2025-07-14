using Strict.Language;
using static Strict.Expressions.MethodExpressionParser;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class MemberCall(Expression? instance, Member member)
	: ConcreteExpression(member.Type, member.IsMutable)
{
	public Expression? Instance { get; } = instance;
	public Member Member { get; } = member;

	public static Expression? TryParse(Body body, Type type, Expression? instance,
		ReadOnlySpan<char> partToParse)
	{
		foreach (var member in type.Members)
			if (partToParse.Equals(member.Name, StringComparison.Ordinal))
				return instance == null && body.IsFakeBodyForMemberInitialization
					? throw new CannotAccessMemberBeforeTypeIsParsed(body, partToParse.ToString(), type)
					: new MemberCall(instance, member);
		return body.Method.Name == Member.ConstraintsBody
			? FindContainingMethodTypeMemberForConstraints(body, instance, partToParse.ToString())
			: null;
	}

	private static MemberCall? FindContainingMethodTypeMemberForConstraints(Body body,
		Expression? instance, string searchFor)
	{
		var member = body.Method.Type.FindMember(searchFor);
		return member != null
			? new MemberCall(instance, member)
			: null;
	}

	public override string ToString() =>
		Instance != null
			? $"{Instance}.{Member.Name}"
			: Member.Name;
}