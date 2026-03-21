using static Strict.Language.Member;

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
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private ExpressionParser parser = null!;

	[Test]
	public void LimitTypeHasCorrectConstants()
	{
		using var limitType = new Type(TestPackage.Instance,
			new TypeLines("Limit",
				"constant MethodLength = 12",
				"constant ParameterCount = 4",
				"constant MethodCount = 15",
				"constant LineCount = 256",
				"constant NestingLevel = 5",
				"constant CharacterCount = 120",
				"constant MultiLineCharacterCount = 100",
				"constant MemberCount = 15",
				"constant MemberCountForEnums = 40",
				"constant NameMaxLimit = 50",
				"constant NameMinLimit = 2")).ParseMembersAndMethods(parser);
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
		Assert.That(limitType.IsEnum, Is.True);
		Assert.That(limitType.Members[0].Name, Is.EqualTo("MethodLength"));
		Assert.That(limitType.Members[0].Type.Name, Is.EqualTo(Type.Number));
		Assert.That(limitType.Members[0].IsConstant, Is.True);
		Assert.That(limitType.Members[^1].Name, Is.EqualTo("NameMinLimit"));
	}

	[Test]
	public void LoadLimitTypeFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var limitLines = new TypeLines("Limit",
			File.ReadAllLines(Path.Combine(langPath, "Limit.strict")));
		using var limitType = new Type(TestPackage.Instance, limitLines).ParseMembersAndMethods(parser);
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
		Assert.That(limitType.IsEnum, Is.True);
	}

	/// <summary>
	/// Keyword.cs has string constants like Has="has", Mutable="mutable", etc.
	/// Most uppercase names (Has, Constant, Let, If, Else, For, With, Return) work as-is.
	/// The one conflict: "Mutable" is a built-in Strict type, so we name it MutableKeyword,
	/// following the same convention as C#'s Type.ValueLowercase for "value".
	/// </summary>
	[Test]
	public void KeywordTypeHasCorrectTextConstants()
	{
		using var keywordType = new Type(TestPackage.Instance,
			new TypeLines("Keyword",
				"constant Has = \"has\"",
				"constant Constant = \"constant\"",
				"constant Let = \"let\"",
				"constant MutableKeyword = \"mutable\"",
				"constant If = \"if\"",
				"constant Else = \"else\"",
				"constant For = \"for\"",
				"constant With = \"with\"",
				"constant Return = \"return\"")).ParseMembersAndMethods(parser);
		Assert.That(keywordType.Members.Count, Is.EqualTo(9));
		Assert.That(keywordType.Members[0].Name, Is.EqualTo("Has"));
		Assert.That(keywordType.Members[0].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(keywordType.Members[3].Name, Is.EqualTo("MutableKeyword"));
		Assert.That(keywordType.Members[3].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(keywordType.Members[3].IsConstant, Is.True);
	}

	[Test]
	public void LoadKeywordTypeFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("Keyword", File.ReadAllLines(Path.Combine(langPath, "Keyword.strict")));
		using var keywordType = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(keywordType.Members.Count, Is.EqualTo(9));
		Assert.That(keywordType.IsEnum, Is.True);
	}

	/// <summary>
	/// TypeKind.cs is an enum with values None, Boolean, Number, etc. In Strict, those names
	/// conflict with built-in types (same issue as C# with System.Type vs Strict.Language.Type).
	/// The workaround: prefix with "Kind" (KindBoolean, KindNumber, etc.) — no conflict, still clear.
	/// </summary>
	[Test]
	public void TypeKindUsesKindPrefixToAvoidBuiltInTypeNameConflicts()
	{
		using var typeKind = new Type(TestPackage.Instance,
			new TypeLines("TypeKind",
				"constant KindNone",
				"constant KindBoolean",
				"constant KindNumber",
				"constant KindText",
				"constant KindCharacter",
				"constant KindList",
				"constant KindDictionary",
				"constant KindError",
				"constant KindEnum",
				"constant KindIterator",
				"constant KindAny",
				"constant KindUnknown")).ParseMembersAndMethods(parser);
		Assert.That(typeKind.IsEnum, Is.True);
		Assert.That(typeKind.Members.Count, Is.EqualTo(12));
		Assert.That(typeKind.Members[0].Name, Is.EqualTo("KindNone"));
		Assert.That(typeKind.Members[0].Type.Name, Is.EqualTo(Type.Number));
		Assert.That(typeKind.Members[1].Name, Is.EqualTo("KindBoolean"));
	}

	[Test]
	public void LoadTypeKindFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("TypeKind",
			File.ReadAllLines(Path.Combine(langPath, "TypeKind.strict")));
		using var typeKind = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(typeKind.Members.Count, Is.EqualTo(12));
		Assert.That(typeKind.IsEnum, Is.True);
	}

	[Test]
	public void UnaryOperatorHasNotConstant()
	{
		using var unaryOp = new Type(TestPackage.Instance,
			new TypeLines("UnaryOperator",
				"constant Not = \"not\"")).ParseMembersAndMethods(parser);
		Assert.That(unaryOp.Members.Count, Is.EqualTo(1));
		Assert.That(unaryOp.Members[0].Name, Is.EqualTo("Not"));
		Assert.That(unaryOp.Members[0].Type.Name, Is.EqualTo(Type.Text));
	}

	[Test]
	public void LoadUnaryOperatorFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("UnaryOperator",
			File.ReadAllLines(Path.Combine(langPath, "UnaryOperator.strict")));
		using var unaryOp = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(unaryOp.Members.Count, Is.EqualTo(1));
		Assert.That(unaryOp.Members[0].Name, Is.EqualTo("Not"));
	}

	private static string GetLanguagePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Language");
}
