using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.CodeValidator;

public sealed record TypeValidator(IEnumerable<Type> Types) : Validator
{
	public void Validate()
	{
		foreach (var type in Types)
		{
			new MethodValidator(type.Methods).Validate();
			new MemberValidator(type).Validate();
			ValidateTypeDependencies(type);
		}
	}

	private static void ValidateTypeDependencies(Type type)
	{
		if (type.Members.Count > 4)
			throw new TypeHasTooManyDependencies(type, type.Members.ToWordList());
	}

	public sealed class TypeHasTooManyDependencies : ParsingFailed
	{
		public TypeHasTooManyDependencies(Type type, string members) : base(type, 0, members) { }
	}
}