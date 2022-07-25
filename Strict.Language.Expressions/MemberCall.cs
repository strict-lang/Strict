using System;
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
				? TryNestedMemberCall(line, partToParse.Split('.'))
				: TryMemberCall(line, partToParse);
	}

	private static MemberCall? TryNestedMemberCall(Method.Line line, SpanSplitEnumerator partsEnumerator) //TODO: use span!
	{
		partsEnumerator.MoveNext();
		var firstMemberName = partsEnumerator.Current.ToString();
		var first = TryMemberCall(line, firstMemberName);
		if (first == null)
			throw new MemberNotFound(line, line.Method.Type, firstMemberName);
		partsEnumerator.MoveNext();
		var secondMemberName = partsEnumerator.Current.ToString();
		var second = first.ReturnType.Members.FirstOrDefault(m => m.Name == secondMemberName);
		return second == null
			? first.ReturnType.Methods.Any(m => m.Name == secondMemberName)
				? null
				: throw new MemberNotFound(line, first.ReturnType, secondMemberName)
			: new MemberCall(first, second);
	}

	private static MemberCall? TryMemberCall(Method.Line line, ReadOnlySpan<char> name)
	{
		if (!name.IsWord())
			return null;
		var memberName = name.ToString(); //TODO: Is this correct? can we still use span after this statement?
		var foundMember = TryLocalMemberCall(line, memberName)
			?? line.Method.Type.Members.FirstOrDefault(member => member.Name == memberName);
		return foundMember != null
			? new MemberCall(foundMember)
			: line.Method.Type.Methods.All(m => m.Name != memberName)
				? throw new MemberNotFound(line, line.Method.Type, memberName)
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