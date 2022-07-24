using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

/// <summary>
/// Links a type member up to an expression making it usable in a method body. Will be written as
/// a local member or if used in a different type via the Type.Member syntax.
/// </summary>
// ReSharper disable once HollowTypeName
public sealed class MemberCall : Expression
{
	public MemberCall(Member member) : base(member.Type) => Member = member;

	public MemberCall(MemberCall parent, Member member) : base(member.Type)
	{
		Parent = parent;
		Member = member;
	}

	public MemberCall? Parent { get; }
	public Member Member { get; }

	public override string ToString() =>
		Parent != null
			? Parent + "." + Member.Name
			: Member.Name;

	public static MemberCall? TryParse(Method.Line line, Range range)
	{
		var partToParse = line.Text.GetSpanFromRange(range);
		return partToParse.Contains('(')
			? null
			: partToParse.Contains('.')
				? TryNestedMemberCall(line, partToParse.ToString().Split('.', 2))//TODO: nononono
				: TryMemberCall(line, partToParse.ToString());//TODO: don't do that
	}

	private static MemberCall? TryNestedMemberCall(Method.Line line, IReadOnlyList<string> parts) //TODO: use span!
	{
		var first = TryMemberCall(line, parts[0]);
		if (first == null)
			throw new MemberNotFound(line, line.Method.Type, parts[0]);
		var second = first.ReturnType.Members.FirstOrDefault(m => m.Name == parts[1]);
		return second == null
			? first.ReturnType.Methods.Any(m => m.Name == parts[1])
				? null
				: throw new MemberNotFound(line, first.ReturnType, parts[1])
			: new MemberCall(first, second);
	}

	private static MemberCall? TryMemberCall(Method.Line line, string name)
	{
		if (!name.IsWord())
			return null;
		var foundMember = TryLocalMemberCall(line, name)
			?? line.Method.Type.Members.FirstOrDefault(member => member.Name == name);
		return foundMember != null
			? new MemberCall(foundMember)
			: line.Method.Type.Methods.All(m => m.Name != name)
				? throw new MemberNotFound(line, line.Method.Type, name)
				: null;
	}

	private static Member? TryLocalMemberCall(Method.Line line, string name) =>
		line.Method.Variables.FirstOrDefault(e => (e as Assignment)?.Name.Name == name) is Assignment
			methodVariable
			? new Member(methodVariable.Name.Name, methodVariable.Value)
			: null;

	public sealed class MemberNotFound : ParsingFailed
	{
		public MemberNotFound(Method.Line line, Type memberType, string memberName) : base(line,
			memberName, memberType) { }
	}
}