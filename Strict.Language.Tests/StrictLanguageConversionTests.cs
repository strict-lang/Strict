using static Strict.Language.Member;

namespace Strict.Language.Tests;

/// <summary>
/// Tests for converting Strict.Language C# types to .strict files.
/// See the Language/ directory at the repository root for the .strict implementations.
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
	}

	[Test]
	public void LoadLimitTypeFromLanguageDirectory()
	{
		var langPath = Path.Combine(
			Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)), "Language");
		var limitLines = new TypeLines("Limit",
			File.ReadAllLines(Path.Combine(langPath, "Limit.strict")));
		using var limitType = new Type(TestPackage.Instance, limitLines).ParseMembersAndMethods(parser);
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
		Assert.That(limitType.IsEnum, Is.True);
	}

	/// <summary>
	/// TypeKind.cs cannot be directly converted to a .strict enum because its constant names
	/// (None, Boolean, Number, Text, etc.) conflict with built-in Strict type names.
	/// This is a blocker: Strict disallows naming a member the same as an existing type unless
	/// the member's type matches the named type.
	/// </summary>
	[Test]
	public void TypeKindEnumConstantsCannotUseBuiltInTypeNamesAsConstants() =>
		Assert.That(
			() => new Type(TestPackage.Instance,
				new TypeLines("TypeKind", "constant None", "constant Boolean")).ParseMembersAndMethods(
				parser),
			Throws.InstanceOf<MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed>());

	/// <summary>
	/// Workaround for TypeKind: use prefixed names (KindBoolean instead of Boolean) to avoid
	/// clashing with built-in type names. This changes the public API but unblocks the conversion.
	/// </summary>
	[Test]
	public void TypeKindCanBeImplementedWithPrefixedNamesAsWorkaround()
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
	}
}
