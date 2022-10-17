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

	public override string ToString() =>
		Instance != null
			? $"{Instance}.{Member.Name}"
			: Member.Name;
}