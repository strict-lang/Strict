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
	public void LanguageStrictFilesMustNotForceToTextInTextContexts()
	{
		var offenders = GetLanguageStrictFiles().
			SelectMany(file => File.ReadAllLines(file).Select((line, lineIndex) =>
				new { file, line, LineNumber = lineIndex + 1 })).
			Where(item => item.line.Contains(" to Text + ", StringComparison.Ordinal) ||
				item.line.Contains("+ ") && item.line.Contains(" to Text", StringComparison.Ordinal) ||
				item.line.Contains(".Log(") && item.line.Contains(" to Text", StringComparison.Ordinal)).
			Select(item => $"{Path.GetFileName(item.file)}:{item.LineNumber}: {item.line.Trim()}");
		Assert.That(offenders, Is.Empty);
	}

	[Test]
	public void ReadLinesBelongsToTextReaderNotFileTrait()
	{
		var root = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		Assert.That(File.ReadAllText(Path.Combine(root, "File.strict")),
			Does.Not.Contain("ReadLines"));
		Assert.That(File.ReadAllText(Path.Combine(root, "TextReader.strict")),
			Does.Contain("ReadLines Texts"));
	}

	[Test]
	public void TypeStrictOwnsTypeParsingSurface()
	{
		var typeSource = File.ReadAllText(Path.Combine(GetLanguagePath(), "Type.strict"));
		Assert.That(typeSource, Does.Contain("Members(lines Texts) Members"));
		Assert.That(typeSource, Does.Contain("Methods(lines Texts) Methods"));
		Assert.That(File.ReadAllText(Path.Combine(GetLanguagePath(), "Parser.strict")),
			Does.Not.Contain("Members(lines Texts"));
	}

	[Test]
	public void BodyStrictParsesLinesIntoConcreteExpressions()
	{
		var bodySource = File.ReadAllText(Path.Combine(GetLanguagePath(), "Body.strict"));
		Assert.That(bodySource, Does.Contain("Expressions ConcreteExpressions"));
		Assert.That(bodySource, Does.Contain("ConcreteExpression"));
		Assert.That(bodySource, Does.Contain("ExpressionKind"));
	}

	[Test]
	public void LanguageStrictFilesAvoidLegacyTextNameFields()
	{
		var forbidden = new[] { "elementName", "typeName", "resultTypeName", "expressionText",
			"typeNames" };
		var offenders = GetLanguageStrictFiles().
			SelectMany(file => File.ReadAllLines(file).Select((line, lineIndex) =>
				new { file, line, LineNumber = lineIndex + 1 })).
			Where(item => item.line.StartsWith("has ", StringComparison.Ordinal) &&
				forbidden.Any(item.line.Contains)).
			Select(item => $"{Path.GetFileName(item.file)}:{item.LineNumber}: {item.line.Trim()}");
		Assert.That(offenders, Is.Empty);
	}

	[Test]
	public void MethodParsingIsSplitFromBaseMethodSignature()
	{
		var root = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		Assert.That(File.ReadAllLines(Path.Combine(root, "Method.strict")),
			Is.EqualTo(new[] { "has Name", "has Type", "has Parameters Variables" }));
		Assert.That(File.Exists(Path.Combine(GetLanguagePath(), "Method.strict")), Is.False);
		var parserSource = File.ReadAllText(Path.Combine(GetLanguagePath(), "MethodParser.strict"));
		Assert.That(parserSource, Does.Contain("Parse(header Text, lines Texts, lineIndex Number) Method"));
		Assert.That(parserSource, Does.Contain("ParseBody(header Text, sourceLines Texts, lineIndex Number) Body"));
	}

	private static IEnumerable<string> GetLanguageStrictFiles() =>
		Directory.GetFiles(GetLanguagePath(), "*.strict");

	[Test]
	public void LoadLimitTypeFromLanguageDirectory()
	{
		using var limitType = CreateLanguageType(TestPackage.Instance, "Limit");
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
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
		AssertInlineTests(limitType, "HasExpectedMethodLimits", 3);
		AssertInlineTests(limitType, "HasExpectedFileLimits", 4);
		AssertInlineTests(limitType, "HasExpectedNameLimits", 4);
	}

	private static Type CreateLanguageType(Package package, string typeName) =>
		new Type(package,
				new TypeLines(typeName,
					File.ReadAllLines(Path.Combine(GetLanguagePath(), $"{typeName}.strict")))).
			ParseMembersAndMethods(new MethodExpressionParser());

	private static string GetLanguagePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Language");

	private static string GetExpressionsPath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Expressions");

	private static void AssertInlineTests(Type type, string methodName, int testCount)
	{
		var method = type.Methods.First(candidate => candidate.Name == methodName);
		method.GetBodyAndParseIfNeeded();
		Assert.That(method.Tests.Count, Is.EqualTo(testCount), methodName);
	}

	[Test]
	public void LoadKeywordTypeFromLanguageDirectory()
	{
		using var keywordType = CreateLanguageType(TestPackage.Instance, "Keyword");
		Assert.That(keywordType.Members.Count, Is.EqualTo(9));
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
		AssertInlineTests(keywordType, "IsKeyword", 4);
		AssertInlineTests(keywordType, "IsDeclarationKeyword", 2);
		AssertInlineTests(keywordType, "IsControlKeyword", 2);
	}

	[Test]
	public void LoadTypeKindFromLanguageDirectory()
	{
		using var typeKind = CreateLanguageType(TestPackage.Instance, "TypeKind");
		Assert.That(typeKind.Members.Count, Is.EqualTo(12));
		Assert.That(typeKind.FindMember("KindNone")!.InitialValue!.ToString(), Is.EqualTo("0"));
		Assert.That(typeKind.FindMember("KindBoolean")!.InitialValue!.ToString(),
			Is.EqualTo(((int)TypeKind.Boolean).ToString()));
		Assert.That(typeKind.FindMember("KindUnknown")!.InitialValue!.ToString(),
			Is.EqualTo(((int)TypeKind.Unknown).ToString()));
		AssertInlineTests(typeKind, "NameOf", 2);
		AssertInlineTests(typeKind, "NameOfScalar", 1);
		AssertInlineTests(typeKind, "NameOfComposite", 1);
		AssertInlineTests(typeKind, "NameOfAny", 2);
	}

	[Test]
	public void LoadUnaryOperatorFromLanguageDirectory()
	{
		using var unaryOperator = CreateLanguageType(TestPackage.Instance, "UnaryOperator");
		Assert.That(unaryOperator.Members.Count, Is.EqualTo(1));
		Assert.That(unaryOperator.FindMember(nameof(UnaryOperator.Not))!.InitialValue!.ToString(),
			Is.EqualTo("\"" + UnaryOperator.Not + "\""));
		AssertInlineTests(unaryOperator, "IsNot", 2);
	}

	[Test]
	public void LoadBinaryOperatorFromLanguageDirectory()
	{
		using var package = new Package(TestPackage.Instance, "BinaryOperatorCheck");
		using var binaryOperator = CreateLanguageType(package, "BinaryOperator");
		Assert.That(binaryOperator.Members.Count, Is.EqualTo(16));
		Assert.That(binaryOperator.IsEnum, Is.True);
		Assert.That(binaryOperator.FindMember(nameof(BinaryOperator.Plus))!.InitialValue!.ToString(),
			Is.EqualTo("\"" + BinaryOperator.Plus + "\""));
		Assert.That(binaryOperator.FindMember(nameof(BinaryOperator.Is))!.InitialValue!.ToString(),
			Is.EqualTo("\"" + BinaryOperator.Is + "\""));
		using var tests = CreateLanguageType(package, "BinaryOperatorTests");
		AssertInlineTests(tests, "Run", 3);
		AssertInlineTests(tests, "ArithmeticSymbols", 6);
		AssertInlineTests(tests, "ComparisonSymbols", 6);
		AssertInlineTests(tests, "LogicalSymbols", 4);
	}

	[Test]
	public async Task LoadTypeParserFromLanguageDirectory()
	{
		using var languagePackage =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Language");
		var typeParser = languagePackage.GetType("Type");
		Assert.That(typeParser.Members.Count, Is.EqualTo(2));
		Assert.That(typeParser.Members[0].Name, Is.EqualTo("Name"));
		Assert.That(typeParser.Members[1].Name, Is.EqualTo("Lines"));
		Assert.That(typeParser.Methods.Count, Is.GreaterThan(10));
		Assert.That(typeParser.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public async Task LoadTypeFinderFromLanguageDirectory()
	{
		using var languagePackage =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Language");
		var typeFinder = languagePackage.GetType("TypeFinder");
		Assert.That(typeFinder.Members.Count, Is.EqualTo(1));
		Assert.That(typeFinder.Members[0].Name, Is.EqualTo("names"));
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
