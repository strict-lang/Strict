namespace Strict.Language.Expressions;

/// <summary>
/// Links a type member up to an expression making it usable in a method body. Will be written as
/// a local member or if used in a different type via the Type.Member syntax.
/// </summary>
// ReSharper disable once HollowTypeName
public sealed class MemberCall : Expression
{
	public MemberCall(Member member) : base(member.Type) => Member = member;
	public Member Member { get; }
	public MemberCall(Expression parent, Member member) : this(member) => Parent = parent;
	public Expression? Parent { get; }

	public override string ToString() =>
		Parent != null
			? Parent + "." + Member.Name
			: Member.Name;
}