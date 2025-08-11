#if TODO
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Validators;

/// <summary>
/// Validator for type members that can work with individual types or use visitor pattern
/// </summary>
public sealed class MemberValidator : Validator
{
	private readonly Type? type;
	private readonly IEnumerable<Member>? members;

	public MemberValidator(Type type)
	{
		this.type = type;
	}

	public MemberValidator(IEnumerable<Member> members)
	{
		this.members = members;
	}

	public void Validate()
	{
		if (type != null)
		{
			var visitor = new MemberValidationVisitor(type);
			visitor.VisitType(type);
		}
		else if (members != null)
		{
			foreach (var member in members)
				ValidateUnusedMember(member, null); // No type context available
		}
	}

	/// <summary>
	/// Visitor that validates members using the visitor pattern
	/// </summary>
	private sealed class MemberValidationVisitor(Type type) : Visitor
	{
		public override void VisitMember(Member member)
		{
			ValidateUnusedMember(member, type);
			
			// Continue with standard traversal
			TryVisitExpression(member.InitialValue);
		}
	}

	private static void ValidateUnusedMember(NamedType member, Type? type)
	{
		if (type != null && type.CountMemberUsage(member.Name) < 2)
			throw new UnusedMemberMustBeRemoved(type, member.Name);
	}

	public sealed class UnusedMemberMustBeRemoved(Type type, string memberName)
		: ParsingFailed(type, 0, memberName);
}
#endif