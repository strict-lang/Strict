using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.CodeValidator.Tests;

public sealed class TypeValidatorTests
{
	[SetUp]
	public void CreatePackageAndParser()
	{
		package = new TestPackage();
		parser = new MethodExpressionParser();
	}

	private Package package = null!;
	private ExpressionParser parser = null!;

	[Test]
	public void ValidateUnusedMember() =>
		Assert.That(
			() => new TypeValidator(new[]
			{
				ParseTypeMethods(CreateType(nameof(ValidateUnusedMember),
					new[]
					{
						"has unused Number",
						"Run(methodInput Number)",
						"\tconstant result = 5 + methodInput",
						"\tresult"
					}))
			}).Validate(), Throws.InstanceOf<MemberValidator.UnusedMemberMustBeRemoved>()!.With.Message.Contains("unused")!);

	private static Type ParseTypeMethods(Type type)
	{
		foreach (var method in type.Methods)
			method.GetBodyAndParseIfNeeded();
		return type;
	}

	private Type CreateType(string typeName, string[] code) =>
		new Type(package, new TypeLines(typeName,
			code)).ParseMembersAndMethods(parser);

	[Test]
	public void ProperlyUsedMemberShouldBeAllowed() =>
		Assert.DoesNotThrow(
			() => new TypeValidator(new[]
			{
				ParseTypeMethods(CreateType(nameof(ProperlyUsedMemberShouldBeAllowed),
					new[]
					{
						"has usedMember Number",
						"Run(methodInput Number)",
						"\tconstant result = usedMember + methodInput",
						"\tresult"
					}))
			}).Validate());
}