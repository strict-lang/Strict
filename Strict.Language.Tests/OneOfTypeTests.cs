namespace Strict.Language.Tests;

public sealed class OneOfTypeTests
{
	[Test]
	public void CreateOneOfType()
	{
		var textOrNumberType =
			new Type(TestPackage.Instance,
					new TypeLines(nameof(CreateOneOfType), "has number", "Run Text or Number",
						"\tif number is 100", "\t\treturn 100", "\t\"It's a text\"")).
				ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(textOrNumberType.Methods[0].ReturnType, Is.InstanceOf<OneOfType>());
		Assert.That(textOrNumberType.Methods[0].ReturnType.Name, Is.EqualTo("TextOrNumber"));
	}
}