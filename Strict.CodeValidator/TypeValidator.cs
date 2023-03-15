using Type = Strict.Language.Type;

namespace Strict.CodeValidator;

public sealed record TypeValidator(IEnumerable<Type> Types) : Validator
{
	public void Validate()
	{
		foreach (var type in Types)
			ValidateType(type);
	}

	private static void ValidateType(Type type)
	{
		new MethodValidator(type.Methods).Validate();
		new MemberValidator(type).Validate();
	}
}