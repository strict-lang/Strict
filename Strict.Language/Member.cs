namespace Strict.Language;

public class Member : NamedType
{
	public Member(Type definedIn, string nameAndType, Expression? value, string keyword = "") : base(definedIn,
		nameAndType, value?.ReturnType)
	{
		if (Type.IsMutable())
			throw new Type.UsingMutableTypesOrImplementsAreNotAllowed(Type, nameAndType);
		Value = value;
		if (keyword == Type.Mutable)
			IsMutable = true;
	}

	public Expression? Value { get; set; }
	public bool IsPublic => char.IsUpper(Name[0]);
}