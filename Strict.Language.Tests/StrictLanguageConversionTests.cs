namespace Strict.Language.Tests;

/// <summary>
/// Tests for converting Strict.Language C# types to .strict files.
/// See the Language/ directory at the repository root for the .strict implementations.
/// Just as C# uses workarounds like Type.ValueLowercase for the "value" keyword or
/// 'using Type = Strict.Language.Type' to avoid System.Type conflicts, Strict uses the
/// same approach: rename conflicting constants (e.g. MutableKeyword, KindBoolean) rather
/// than treating conflicts as blockers.
/// </summary>
public sealed class StrictLanguageConversionTests
{
	[Test]
	public void LoadLimitTypeFromLanguageDirectory()
	{
		using var limitType = CreateLanguageType(TestPackage.Instance, "Limit");
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
		Assert.That(limitType.IsEnum, Is.True);
		Assert.That(limitType.FindMember(nameof(Limit.MethodLength))!.InitialValue!.ToString(),
			Is.EqualTo(Limit.MethodLength.ToString()));
		Assert.That(limitType.FindMember(nameof(Limit.CharacterCount))!.InitialValue!.ToString(),
			Is.EqualTo(Limit.CharacterCount.ToString()));
		Assert.That(limitType.FindMember(nameof(Limit.LineCount))!.InitialValue!.ToString(),
			Is.EqualTo(Limit.LineCount.ToString()));
		Assert.That(limitType.FindMember(nameof(Limit.MethodCount))!.InitialValue!.ToString(),
			Is.EqualTo(Limit.MethodCount.ToString()));
		Assert.That(limitType.FindMember(nameof(Limit.NestingLevel))!.InitialValue!.ToString(),
			Is.EqualTo(Limit.NestingLevel.ToString()));
		Assert.That(limitType.FindMember(nameof(Limit.ParameterCount))!.InitialValue!.ToString(),
			Is.EqualTo(Limit.ParameterCount.ToString()));
	}

	private static Type CreateLanguageType(Package package, string typeName) =>
		new Type(package,
				new TypeLines(typeName,
					File.ReadAllLines(Path.Combine(GetLanguagePath(), $"{typeName}.strict")))).
			ParseMembersAndMethods(new MethodExpressionParser());

	private static string GetLanguagePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Language");

	[Test]
	public void LoadKeywordTypeFromLanguageDirectory()
	{
		using var keywordType = CreateLanguageType(TestPackage.Instance, "Keyword");
		Assert.That(keywordType.Members.Count, Is.EqualTo(9));
		Assert.That(keywordType.IsEnum, Is.True);
		Assert.That(keywordType.FindMember(nameof(Keyword.Constant))!.InitialValue!.ToString(),
			Is.EqualTo("\"" + Keyword.Constant + "\""));
		Assert.That(keywordType.FindMember(nameof(Keyword.For))!.InitialValue!.ToString(),
			Is.EqualTo("\"" + Keyword.For + "\""));
		Assert.That(keywordType.FindMember(nameof(Keyword.Return))!.InitialValue!.ToString(),
			Is.EqualTo("\"" + Keyword.Return + "\""));
		Assert.That(keywordType.FindMember(nameof(Keyword.Let))!.InitialValue!.ToString(),
			Is.EqualTo("\"" + Keyword.Let + "\""));
		Assert.That(
			keywordType.FindMember(nameof(Keyword.Mutable) + "Keyword")!.InitialValue!.ToString(),
			Is.EqualTo("\"" + Keyword.Mutable + "\""));
	}

	[Test]
	public void LoadTypeParserFromLanguageDirectory()
	{
		using var languagePackage = new Package(TestPackage.Instance, "Language");
		using var typeParser = CreateLanguageType(languagePackage, "Type");
		Assert.That(typeParser.Members.Count, Is.EqualTo(1));
		Assert.That(typeParser.Members[0].Name, Is.EqualTo("lines"));
		Assert.That(typeParser.Methods.Count, Is.EqualTo(2));
		Assert.That(typeParser.Methods[0].Name, Is.EqualTo("IsMember"));
		Assert.That(typeParser.Methods[1].Name, Is.EqualTo("IsMethodHeader"));
	}

	[Test]
	public void LoadTypeFinderFromLanguageDirectory()
	{
		using var languagePackage = new Package(TestPackage.Instance, "Language");
		using var typeFinder = CreateLanguageType(languagePackage, "TypeFinder");
		Assert.That(typeFinder.Members.Count, Is.EqualTo(1));
		Assert.That(typeFinder.Members[0].Name, Is.EqualTo("typeNames"));
		Assert.That(typeFinder.Methods.Count, Is.EqualTo(1));
		Assert.That(typeFinder.Methods[0].Name, Is.EqualTo("Count"));
	}

	[Test]
	public void LoadContextFromLanguageDirectory()
	{
		using var languagePackage = new Package(TestPackage.Instance, "Language");
		using var contextType = CreateLanguageType(languagePackage, "Context");
		Assert.That(contextType.Members.Count, Is.EqualTo(4));
		Assert.That(contextType.Members[0].Name, Is.EqualTo("Parent"));
		Assert.That(contextType.Members[1].Name, Is.EqualTo("Name"));
		Assert.That(contextType.Members[2].Name, Is.EqualTo("FullName"));
		Assert.That(contextType.FindMember("ParentSeparator")!.InitialValue!.ToString(),
			Is.EqualTo("\"/\""));
	}
}