using Strict.Language;
using static Strict.Expressions.MethodExpressionParser;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class MemberCall(Expression? instance, Member member, int lineNumber = 0,
	bool shouldUseTypePrefixWithoutInstance = false)
	: ConcreteExpression(member.Type, lineNumber, member.IsMutable)
{
	public Expression? Instance { get; } = instance;
	public Member Member { get; } = member;
	private bool ShouldUseTypePrefixWithoutInstance { get; } = shouldUseTypePrefixWithoutInstance;

	public static Expression? TryParse(Body body, Type type, Expression? instance,
		ReadOnlySpan<char> partToParse)
	{
		var members = type.Members;
		for (var i = 0; i < members.Count; i++)
			if (partToParse.Equals(members[i].Name, StringComparison.Ordinal))
			{
				if (!members[i].IsConstant && instance == null && body.IsFakeBodyForMemberInitialization)
					throw new CannotAccessMemberBeforeTypeIsParsed(body, partToParse.ToString(), type);
				var member = members[i];
				if (type.IsGeneric && type.IsList && member.Type.IsGeneric && member.Type.IsList)
					member = member.CloneWithImplementation(
						type.GetGenericImplementation(type.GetType(Type.GenericUppercase)));
				else if (member.Type == type && type.IsGeneric)
					member = member.CloneWithImplementation(
						type.GetGenericImplementation(type.GetType(Type.GenericUppercase)));
				return new MemberCall(instance, member, body.CurrentFileLineNumber,
					instance == null && type != body.Method.Type);
			}
		return body.Method.Name == Member.ConstraintsBody
			? FindContainingMethodTypeMemberForConstraints(body, instance, partToParse.ToString())
			: null;
	}

	private static MemberCall? FindContainingMethodTypeMemberForConstraints(Body body,
		Expression? instance, string searchFor)
	{
		var member = body.Method.Type.FindMember(searchFor) ??
			body.Method.ConstraintDeclaringType?.FindMember(searchFor);
		return member != null
			? new MemberCall(instance, member, body.CurrentFileLineNumber)
			: null;
	}

	public override bool IsConstant => (Instance?.IsConstant ?? true) && Member.IsConstant;

	public override string ToString() =>
		Instance != null
			? $"{Instance}.{Member.Name}"
			: ShouldUseTypePrefixWithoutInstance
				? $"{Member.DefinedIn.Name}.{Member.Name}"
				: Member.Name;

	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) || other is MemberCall mc && Member.Name == mc.Member.Name &&
		Member.Type == mc.Member.Type && Equals(Instance, mc.Instance);

	public override int GetHashCode() =>
		Member.GetHashCode() ^ (Instance?.GetHashCode() ?? 0);
}