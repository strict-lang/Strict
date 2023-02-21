using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.CodeValidator.Tests;

public sealed class TypeValidatorTests
{
	[Test]
	public void ValidateUnusedMember()
	{
		var type = new Type(new TestPackage(), new TypeLines(nameof(ValidateUnusedMember), ""));
		var typeValidator = new TypeValidator(new[] { type });
	}
}