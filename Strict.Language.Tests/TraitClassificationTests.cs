namespace Strict.Language.Tests;

public sealed class TraitClassificationTests
{
	[Test]
	public void TraitCanComposeOtherTraitsWithPublicHasMembers()
	{
		using var package = new Package(nameof(TraitCanComposeOtherTraitsWithPublicHasMembers));
		var parser = new MethodExpressionParser();
		var readable = new Type(package, new TypeLines("Readable", "Read"));
		var writable = new Type(package, new TypeLines("Writable", "Write"));
		using var file = new Type(package, new TypeLines("FileLike",
			"has Readable",
			"has Writable",
			"Open"));
		foreach (var type in new[] { readable, writable, file })
			type.ParseMembersAndMethods(parser);
		Assert.That(file.IsTrait, Is.True);
	}

	[Test]
	public void TypeWithLowercaseTraitDependencyCanHaveMethodBody()
	{
		using var package = new Package(nameof(TypeWithLowercaseTraitDependencyCanHaveMethodBody));
		var parser = new MethodExpressionParser();
		var writable = new Type(package, new TypeLines("Writable", "Write"));
		using var logger = new Type(package, new TypeLines("LoggerLike",
			"has writable",
			"Log",
			"\twritable.Write"));
		writable.ParseMembersAndMethods(parser);
		logger.ParseMembersAndMethods(parser);
		Assert.That(logger.IsTrait, Is.False);
	}
}
