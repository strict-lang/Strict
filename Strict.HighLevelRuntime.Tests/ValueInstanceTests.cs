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
		Assert.That(ValueInstance.CreateNumber(TestPackage.Instance, 42d).ToString(),
			Is.EqualTo("Number:42"));

	[Test]
	public void CompareTwoNumbers() =>
		Assert.That(ValueInstance.CreateNumber(TestPackage.Instance, 42d),
			Is.EqualTo(ValueInstance.CreateNumber(TestPackage.Instance, 42d)));

	[Test]
	public void CompareNumberToText() =>
		Assert.That(ValueInstance.CreateNumber(TestPackage.Instance, 5d),
			Is.Not.EqualTo(ValueInstance.CreateText(TestPackage.Instance, "5")));

	[Test]
	public void CompareLists()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var nums1 = new List<ValueInstance>
		{
			ValueInstance.CreateNumber(TestPackage.Instance, 1),
			ValueInstance.CreateNumber(TestPackage.Instance, 2),
			ValueInstance.CreateNumber(TestPackage.Instance, 3)
		};
		var nums2 = new List<ValueInstance>
		{
			ValueInstance.CreateNumber(TestPackage.Instance, 1),
			ValueInstance.CreateNumber(TestPackage.Instance, 2),
			ValueInstance.CreateNumber(TestPackage.Instance, 3)
		};
		var nums3 = new List<ValueInstance>
		{
			ValueInstance.CreateNumber(TestPackage.Instance, 1),
			ValueInstance.CreateNumber(TestPackage.Instance, 2),
			ValueInstance.CreateNumber(TestPackage.Instance, 1)
		};
		var list = ValueInstance.CreateObject(listType, nums1);
		Assert.That(list, Is.EqualTo(ValueInstance.CreateObject(listType, nums2)));
		Assert.That(list, Is.Not.EqualTo(ValueInstance.CreateObject(listType, nums3)));
	}

	[Test]
	public void ListWithExpressionsThrows()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var expressions = new Expression[] { new Value(numberType, 1) };
		Assert.Throws<ValueInstance.InvalidTypeValue>(() =>
			ValueInstance.CreateObject(listType, expressions));
	}

	[Test]
	public void GenericListTypeAcceptsValueInstances()
	{
		var listType = TestPackage.Instance.GetType("List(key Generic, mappedValue Generic)");
		Assert.That(ValueInstance.CreateObject(listType, new List<ValueInstance>()), Is.Not.Null);
	}

	[Test]
	public void CompareDictionaries()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var d1 = new Dictionary<ValueInstance, ValueInstance>
		{
			{ ValueInstance.CreateNumber(TestPackage.Instance, 1), ValueInstance.CreateNumber(TestPackage.Instance, 2) }
		};
		var d2 = new Dictionary<ValueInstance, ValueInstance>
		{
			{ ValueInstance.CreateNumber(TestPackage.Instance, 1), ValueInstance.CreateNumber(TestPackage.Instance, 2) }
		};
		var d3 = new Dictionary<ValueInstance, ValueInstance>
		{
			{ ValueInstance.CreateNumber(TestPackage.Instance, 2), ValueInstance.CreateNumber(TestPackage.Instance, 2) }
		};
		var d4 = new Dictionary<ValueInstance, ValueInstance>
		{
			{ ValueInstance.CreateNumber(TestPackage.Instance, 1), ValueInstance.CreateNumber(TestPackage.Instance, 3) }
		};
		var d5 = new Dictionary<ValueInstance, ValueInstance>
		{
			{ ValueInstance.CreateNumber(TestPackage.Instance, 1), ValueInstance.CreateNumber(TestPackage.Instance, 3) },
			{ ValueInstance.CreateNumber(TestPackage.Instance, 2), ValueInstance.CreateNumber(TestPackage.Instance, 2) }
		};
		var list = ValueInstance.CreateObject(dictType, d1);
		Assert.That(list, Is.EqualTo(ValueInstance.CreateObject(dictType, d2)));
		Assert.That(list, Is.Not.EqualTo(ValueInstance.CreateObject(dictType, d3)));
		Assert.That(list, Is.Not.EqualTo(ValueInstance.CreateObject(dictType, d4)));
		Assert.That(list, Is.Not.EqualTo(ValueInstance.CreateObject(dictType, d5)));
	}

	[Test]
	public void CompareTypeContainingNumber()
	{
		using var t =
			new Type(TestPackage.Instance,
				new TypeLines(nameof(CompareTypeContainingNumber), "has number", "Run Boolean",
					"\tnumber is 42")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(ValueInstance.Create(t, 42d),
			Is.EqualTo(ValueInstance.CreateNumber(TestPackage.Instance, 42d)));
	}

	[Test]
	public void NoneTypeRejectsNonNullValue()
	{
		var package = new Package(nameof(NoneTypeRejectsNonNullValue));
		var noneType = package.FindType(Base.None, package);
		Assert.That(() => ValueInstance.Create(noneType!, 1),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void NonNoneTypeRejectsNullValue() =>
		Assert.That(() => ValueInstance.CreateNone(numberType.Package),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());

	[Test]
	public void BooleanTypeRejectsNonBoolValue()
	{
		var boolType = TestPackage.Instance.GetType(Base.Boolean);
		Assert.That(() => ValueInstance.Create(boolType, 1),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void ObjectConstructorRejectsBooleanType()
	{
		var boolType = TestPackage.Instance.GetType(Base.Boolean);
		Assert.That(() => ValueInstance.CreateObject(boolType, (object)true),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void EnumTypeAllowsNumberAndText()
	{
		var enumType = new Type(TestPackage.Instance,
				new TypeLines(nameof(EnumTypeAllowsNumberAndText), "constant One",
					"constant Something = \"Something\"")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => ValueInstance.Create(enumType, 1), Throws.Nothing);
	}

	[Test]
	public void EnumTypeRejectsNonEnumValue()
	{
		var enumType = new Type(TestPackage.Instance,
				new TypeLines(nameof(EnumTypeRejectsNonEnumValue), "constant One")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => ValueInstance.Create(enumType, true),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void TextTypeRejectsNonStringValue()
	{
		var textType = TestPackage.Instance.GetType(Base.Text);
		Assert.That(() => ValueInstance.Create(textType, 5),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void ObjectConstructorRejectsTextType()
	{
		var textType = TestPackage.Instance.GetType(Base.Text);
		Assert.That(() => ValueInstance.CreateObject(textType, (object)"text"),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void CharacterTypeRejectsNonCharacterValue()
	{
		var charType = TestPackage.Instance.GetType(Base.Character);
		Assert.That(() => ValueInstance.Create(charType, "A"),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void ListTypeRejectsInvalidValue()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		Assert.That(() => ValueInstance.CreateObject(listType, new object()),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void NumberTypeRejectsNonNumericValue() =>
		Assert.That(() => ValueInstance.Create(numberType, "nope"),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());

	[Test]
	public void ObjectConstructorRejectsNumberType() =>
		Assert.That(() => ValueInstance.CreateObject(numberType, (object)1d),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());

	[Test]
	public void ObjectConstructorRejectsNoneType()
	{
		var package = new Package(nameof(ObjectConstructorRejectsNoneType));
		var noneType = package.FindType(Base.None, package);
		Assert.That(() => ValueInstance.CreateObject(noneType!, (object?)null),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}

	[Test]
	public void TypeRejectsUnknownDictionaryMembers()
	{
		using var t = new Type(TestPackage.Instance,
				new TypeLines(nameof(TypeRejectsUnknownDictionaryMembers), "has number", "has text")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var values = new Dictionary<string, object?> { { "wrong", 1 } };
		Assert.That(() => ValueInstance.CreateObject(t, values),
			Throws.InstanceOf<ValueInstance.UnableToAssignMemberToType>());
	}

	[Test]
	public void NonErrorTypeRejectsUnsupportedValue()
	{
		using var t = new Type(TestPackage.Instance,
				new TypeLines(nameof(NonErrorTypeRejectsUnsupportedValue), "has number", "has text")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => ValueInstance.CreateObject(t, new object()),
			Throws.InstanceOf<ValueInstance.InvalidTypeValue>());
	}
}