using Type = Strict.Language.Type;

namespace Strict.CodeValidator;

public sealed record TypeValidator(IEnumerable<Type> Types) : Validator
{
	public void Validate()
	{

	}
}