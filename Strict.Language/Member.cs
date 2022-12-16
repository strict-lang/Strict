namespace Strict.Language;

public class Member : NamedType
{
	public Member(Type definedIn, string nameAndType, Expression? value, bool usedMutableKeyword = false) : base(definedIn,
		nameAndType, value?.ReturnType)
	{
		Value = value;
		if (usedMutableKeyword)
			IsMutable = true;
	}

	public Expression? Value { get; set; }
	public bool IsPublic => char.IsUpper(Name[0]);
}