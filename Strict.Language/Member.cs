namespace Strict.Language;

public class Member : NamedType
{
	public Member(Type definedIn, string nameAndType, Expression? value) : base(definedIn,
		nameAndType, value?.ReturnType) =>
		Value = value;

	public Expression? Value { get; }
	public bool IsPublic => char.IsUpper(Name[0]);
}