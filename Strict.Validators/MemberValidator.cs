using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Validators;

public sealed record MemberValidator(Type Type) : Validator
{
	public void Validate()
	{
		foreach (var member in Type.Members)
			ValidateUnusedMember(member);
	}

	private void ValidateUnusedMember(NamedType member)
	{
		if (Type.CountMemberUsage(member.Name) < 2)
			throw new UnusedMemberMustBeRemoved(Type, member.Name);
	}

	public sealed class UnusedMemberMustBeRemoved(Type type, string memberName)
		: ParsingFailed(type, 0, memberName);
}