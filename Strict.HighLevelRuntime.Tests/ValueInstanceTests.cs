using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ValueInstanceTests
{
	[SetUp]
	public void CreateNumber() => numberType = TestPackage.Instance.GetType(Base.Number);

	private Type numberType = null!;

	[Test]
	public void ToStringShowsTypeAndValue() =>
		Assert.That(new ValueInstance(numberType, 42).ToString(), Is.EqualTo("Number:42"));

	[Test]
	public void CompareTwoNumbers() =>
		Assert.That(new ValueInstance(numberType, 42), Is.EqualTo(new ValueInstance(numberType, 42)));

	[Test]
	public void CompareNumberToText() =>
		Assert.That(new ValueInstance(numberType, 5),
			Is.Not.EqualTo(new ValueInstance(TestPackage.Instance.GetType(Base.Text), "5")));

	[Test]
	public void CompareLists()
	{
		var list = new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
			new[] { 1, 2, 3 });
		Assert.That(list,
			Is.EqualTo(new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
				new[] { 1, 2, 3 })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
				new[] { 1, 2, 1 })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
				new[] { 1, 2, 3, 4 })));
	}

	[Test]
	public void ListWithExpressionsThrows()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var expressions = new Expression[] { new Value(numberType, 1) };
		Assert.Throws<ValueInstance.InvalidTypeValue>(() => new ValueInstance(listType, expressions));
	}

	[Test]
	public void GenericListTypeAcceptsValueInstances()
	{
		var listType = TestPackage.Instance.GetType("List(key Generic, mappedValue Generic)");
		Assert.That(new ValueInstance(listType, new List<ValueInstance>()), Is.Not.Null);
	}

	[Test]
	public void CompareDictionaries()
	{
		var list = new ValueInstance(
			TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
			new Dictionary<int, int> { { 1, 2 } });
		Assert.That(list,
			Is.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 1, 2 } })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 2, 2 } })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 1, 3 } })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 1, 3 }, { 2, 2 } })));
	}

	[Test]
	public void CompareTypeContainingNumber()
	{
		using var t =
			new Type(TestPackage.Instance,
				new TypeLines(nameof(CompareTypeContainingNumber), "has number", "Run Boolean",
					"\tnumber is 42")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(new ValueInstance(t, 42), Is.EqualTo(new ValueInstance(numberType, 42)));
	}

	[Test]
	public void NoneTypeRejectsNonNullValue()
	{
		var package = new Package(nameof(NoneTypeRejectsNonNullValue));
		var noneType = package.FindType(Base.None, package);
		Assert.That(() => new ValueInstance(noneType!, 1),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void NonNoneTypeRejectsNullValue() =>
		Assert.That(() => new ValueInstance(numberType, null),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());

	[Test]
	public void BooleanTypeRejectsNonBoolValue()
	{
		var boolType = TestPackage.Instance.GetType(Base.Boolean);
		Assert.That(() => new ValueInstance(boolType, 1),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void EnumTypeAllowsNumberAndText()
	{
		var enumType = new Type(TestPackage.Instance,
				new TypeLines(nameof(EnumTypeAllowsNumberAndText), "constant One", "constant Something = \"Something\"")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => new ValueInstance(enumType, 1), Throws.Nothing);
	}

	[Test]
	public void EnumTypeRejectsNonEnumValue()
	{
		var enumType = new Type(TestPackage.Instance,
				new TypeLines(nameof(EnumTypeRejectsNonEnumValue), "constant One")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => new ValueInstance(enumType, true),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void TextTypeRejectsNonStringValue()
	{
		var textType = TestPackage.Instance.GetType(Base.Text);
		Assert.That(() => new ValueInstance(textType, 5),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void CharacterTypeRejectsNonCharacterValue()
	{
		var charType = TestPackage.Instance.GetType(Base.Character);
		Assert.That(() => new ValueInstance(charType, "A"),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void ListTypeRejectsInvalidValue()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		Assert.That(() => new ValueInstance(listType, new object()),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void NumberTypeRejectsNonNumericValue() =>
		Assert.That(() => new ValueInstance(numberType, "nope"),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());

	[Test]
	public void TypeRejectsUnknownDictionaryMembers()
	{
		using var t = new Type(TestPackage.Instance,
				new TypeLines(nameof(TypeRejectsUnknownDictionaryMembers), "has number", "has text")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var values = new Dictionary<string, object?> { { "wrong", 1 } };
		Assert.That(() => new ValueInstance(t, values),
			Throws.InstanceOf<ValueInstance.UnableToAssignMemberToType>());
	}

	[Test]
	public void NonErrorTypeRejectsUnsupportedValue()
	{
		using var t = new Type(TestPackage.Instance,
				new TypeLines(nameof(NonErrorTypeRejectsUnsupportedValue), "has number", "has text")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => new ValueInstance(t, new object()),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}
}