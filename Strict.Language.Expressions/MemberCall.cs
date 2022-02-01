using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

/// <summary>
/// Links a type member up to an expression making it usable in a method body. Will be written as
/// a local member or if used in a different type via the Type.Member syntax.
/// </summary>
// ReSharper disable once HollowTypeName
public class MemberCall : Expression
{
	public MemberCall(Member member) : base(member.Type) => Member = member;

	public MemberCall(MemberCall parent, Member member) : base(member.Type)
	{
		this.parent = parent;
		Member = member;
	}

	private readonly MemberCall? parent;
	public Member Member { get; }

	public override string ToString() =>
		parent != null
			? parent + "." + Member.Name
			: Member.Name;

	public static MemberCall? TryParse(Method context, string input) =>
		input.Contains('.')
			? TryNestedMemberCall(context, input.Split('.', 2))
			: TryMemberCall(context, input);

	private static MemberCall TryNestedMemberCall(Method context, IReadOnlyList<string> parts)
	{
		var first = TryMemberCall(context, parts[0])!;
		var second = first.ReturnType.Members.FirstOrDefault(m => m.Name == parts[1]);
		return second == null
			? throw new MemberNotFound(parts[1], first.ReturnType)
			: new MemberCall(first, second);
	}

	private static MemberCall? TryMemberCall(Method context, string name)
	{
		if (!name.IsWord())
			return null;
		var foundMember = context.Type.Members.FirstOrDefault(member => member.Name == name);
		if (foundMember == null)
			throw new MemberNotFound(name, context.Type);
		return new MemberCall(foundMember);
	}

	public class MemberNotFound : Exception
	{
		public MemberNotFound(string memberName, Type type) : base(memberName + " in " + type) { }
	}
}