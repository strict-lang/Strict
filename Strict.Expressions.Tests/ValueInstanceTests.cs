using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class ValueInstanceTests
{
	[SetUp]
	public void CreateNumber() => numberType = TestPackage.Instance.GetType(Type.Number);

	private Type numberType = null!;

	[Test]
	public void ToStringShowsTypeAndValue() =>
		Assert.That(new ValueInstance(numberType, 42d).ToString(), Is.EqualTo("Number: 42"));

	[Test]
	public void CompareTwoNumbers() =>
		Assert.That(new ValueInstance(numberType, 42d),
			Is.EqualTo(new ValueInstance(numberType, 42d)));

	[Test]
	public void CompareNumberToText() =>
		Assert.That(new ValueInstance(numberType, 5d),
			Is.Not.EqualTo(new ValueInstance("5")));

	[Test]
	public void CompareLists()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var nums1 = new List<ValueInstance>
		{
			new(numberType, 1d),
			new(numberType, 2d),
			new(numberType, 3d)
		};
		var nums2 = new List<ValueInstance>
		{
			new(numberType, 1d),
			new(numberType, 2d),
			new(numberType, 3d)
		};
		var nums3 = new List<ValueInstance>
		{
			new(numberType, 1d),
			new(numberType, 2d),
			new(numberType, 1d)
		};
		var list = new ValueInstance(listType, nums1);
		Assert.That(list, Is.EqualTo(new ValueInstance(listType, nums2)));
		Assert.That(list, Is.Not.EqualTo(new ValueInstance(listType, nums3)));
	}

	[Test]
	public void ListWithValueInstancesWorks()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var items = new List<ValueInstance> { new Number(numberType, 1d).Data };
		Assert.That(new ValueInstance(listType, items), Is.Not.Null);
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
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var k1 = new ValueInstance(numberType, 1d);
		var k2 = new ValueInstance(numberType, 2d);
		var v2 = new ValueInstance(numberType, 2d);
		var v3 = new ValueInstance(numberType, 3d);
		var d1 = new Dictionary<ValueInstance, ValueInstance> { { k1, v2 } };
		var d2 = new Dictionary<ValueInstance, ValueInstance> { { new ValueInstance(numberType, 1d), new ValueInstance(numberType, 2d) } };
		var d3 = new Dictionary<ValueInstance, ValueInstance> { { k2, v2 } };
		var d4 = new Dictionary<ValueInstance, ValueInstance> { { k1, v3 } };
		var d5 = new Dictionary<ValueInstance, ValueInstance>
		{
			{ k1, v3 },
			{ k2, v2 }
		};
		var list = new ValueInstance(dictType, d1);
		Assert.That(list, Is.EqualTo(new ValueInstance(dictType, d2)));
		Assert.That(list, Is.Not.EqualTo(new ValueInstance(dictType, d3)));
		Assert.That(list, Is.Not.EqualTo(new ValueInstance(dictType, d4)));
		Assert.That(list, Is.Not.EqualTo(new ValueInstance(dictType, d5)));
	}

	[Test]
	public void CompareTypeContainingNumber()
	{
		using var t =
			new Type(TestPackage.Instance,
				new TypeLines(nameof(CompareTypeContainingNumber), "has number", "Run Boolean",
					"\tnumber is 42")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(new ValueInstance(t, 42d), Is.EqualTo(new ValueInstance(numberType, 42d)));
	}

	[Test]
	public void ValueListInstanceStoresTypeAndItems()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var items = new List<ValueInstance> { new(numberType, 1d), new(numberType, 2d) };
		var instance = new ValueInstance(listType, items);
		Assert.That(instance.ToString(), Does.Contain("List"));
	}

	[Test]
	public void ValueDictionaryInstanceStoresTypeAndItems()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var dict = new Dictionary<ValueInstance, ValueInstance>
		{
			{ new ValueInstance(numberType, 1), new ValueInstance(numberType, 2) }
		};
		var instance = new ValueInstance(dictType, dict);
		Assert.That(instance.ToString(), Does.Contain("Dictionary"));
	}

	[Test]
	public void ValueTypeInstanceStoresMembers()
	{
		using var t = new Type(TestPackage.Instance,
			new TypeLines(nameof(ValueTypeInstanceStoresMembers), "has number", "Run Boolean",
				"\tnumber is 1")).ParseMembersAndMethods(new MethodExpressionParser());
		var instance = new ValueInstance(t,
			new Dictionary<string, ValueInstance> { { "number", new ValueInstance(numberType, 7) } });
		Assert.That(instance.ToString(), Does.Contain(nameof(ValueTypeInstanceStoresMembers)));
	}

	[Test]
	public void ThrowsInvalidTypeValueWhenUsingReservedNumberForText() =>
		Assert.Throws<ValueInstance.InvalidTypeValue>(() =>
			_ = new ValueInstance(numberType, -7.90897526e307));

	[Test]
	public void ThrowsWhenCreatingTypeInstanceForNumberType() =>
		Assert.Throws<ValueInstance.ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes>(() =>
			_ = new ValueInstance(numberType, new Dictionary<string, ValueInstance>()));

	[Test]
	public void CopyConstructorWithDictionaryCreatesNewReturnType()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var mutableDictType =
			TestPackage.Instance.GetType(Type.Mutable).GetGenericImplementation(dictType);
		var original = new ValueInstance(dictType,
			new Dictionary<ValueInstance, ValueInstance>
			{
				{ new ValueInstance(numberType, 1d), new ValueInstance(numberType, 2d) }
			});
		var copy = new ValueInstance(original, mutableDictType);
		Assert.That(copy.IsDictionary, Is.True);
		Assert.That(copy.IsMutable, Is.True);
	}

	[Test]
	public void CopyConstructorWithMutableTextTypeIdConvertsBackToTextId()
	{
		var textType = TestPackage.Instance.GetType(Type.Text);
		var mutableTextType =
			TestPackage.Instance.GetType(Type.Mutable).GetGenericImplementation(textType);
		var textInstance = new ValueInstance("hello");
		var mutableTextInstance = new ValueInstance(textInstance, mutableTextType);
		var converted = new ValueInstance(mutableTextInstance, textType);
		Assert.That(converted.IsText, Is.True);
		Assert.That(converted.Text, Is.EqualTo("hello"));
	}

	[Test]
	public void CopyConstructorWithTypeIdCreatesNewReturnType()
	{
		using var originalType = new Type(TestPackage.Instance,
			new TypeLines(nameof(CopyConstructorWithTypeIdCreatesNewReturnType) + "A", "has number",
				"Run Boolean", "\tnumber is 1")).ParseMembersAndMethods(new MethodExpressionParser());
		using var newType = new Type(TestPackage.Instance,
			new TypeLines(nameof(CopyConstructorWithTypeIdCreatesNewReturnType) + "B", "has number",
				"Run Boolean", "\tnumber is 1")).ParseMembersAndMethods(new MethodExpressionParser());
		var members = new Dictionary<string, ValueInstance>
		{
			{ "number", new ValueInstance(numberType, 5d) }
		};
		var original = new ValueInstance(originalType, members);
		var copy = new ValueInstance(original, newType);
		Assert.That(copy.TryGetValueTypeInstance()!.ReturnType, Is.EqualTo(newType));
	}

	[Test]
	public void IsTypeReturnsTrueForDictionaryInstanceMatchingType()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var instance = new ValueInstance(dictType, new Dictionary<ValueInstance, ValueInstance>());
		Assert.That(instance.IsType(dictType), Is.True);
		Assert.That(instance.IsType(numberType), Is.False);
	}

	[Test]
	public void ApplyMethodReturnTypeMutableConvertsFromMutableToImmutable()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var mutableListType =
			TestPackage.Instance.GetType(Type.Mutable).GetGenericImplementation(listType);
		var mutableInstance =
			new ValueInstance(mutableListType, new List<ValueInstance> { new(numberType, 1d) });
		var result = mutableInstance.ApplyMethodReturnTypeMutable(listType);
		Assert.That(result.IsMutable, Is.False);
		Assert.That(result.IsList, Is.True);
	}

	[Test]
	public void GetTypeExceptTextReturnsListReturnTypeForListInstance()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var instance = new ValueInstance(listType, new List<ValueInstance>());
		Assert.That(instance.GetTypeExceptText(), Is.EqualTo(listType));
	}

	[Test]
	public void GetIteratorLengthThrowsForDictionaryInstance()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var instance = new ValueInstance(dictType, new Dictionary<ValueInstance, ValueInstance>());
		Assert.Throws<ValueInstance.IteratorNotSupported>(() => _ = instance.GetIteratorLength());
	}

	[Test]
	public void GetIteratorLengthForTypeIdWithKeysAndValuesMember()
	{
		using var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(GetIteratorLengthForTypeIdWithKeysAndValuesMember), "has number",
				"Run Number", "\t5")).ParseMembersAndMethods(new MethodExpressionParser());
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var listInstance = new ValueInstance(listType,
			new List<ValueInstance> { new(numberType, 1d), new(numberType, 2d), new(numberType, 3d) });
		var members = new Dictionary<string, ValueInstance>
		{
			{ "number", new ValueInstance(numberType, 1d) },
			{ "keysAndValues", listInstance }
		};
		var instance = new ValueInstance(customType, members);
		Assert.That(instance.GetIteratorLength(), Is.EqualTo(3));
	}

	[Test]
	public void GetIteratorValueForTypeIdWithElementsMember()
	{
		using var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(GetIteratorValueForTypeIdWithElementsMember), "has number",
				"Run Number", "\t5")).ParseMembersAndMethods(new MethodExpressionParser());
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var item1 = new ValueInstance(numberType, 10d);
		var item2 = new ValueInstance(numberType, 20d);
		var listInstance = new ValueInstance(listType, new List<ValueInstance> { item1, item2 });
		var members = new Dictionary<string, ValueInstance>
		{
			{ "number", new ValueInstance(numberType, 1d) },
			{ "elements", listInstance }
		};
		var instance = new ValueInstance(customType, members);
		var charType = TestPackage.Instance.GetType(Type.Character);
		Assert.That(instance.GetIteratorValue(charType, 1), Is.EqualTo(item2));
	}

	[Test]
	public void GetIteratorValueThrowsForUnsupportedInstance()
	{
		using var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(GetIteratorValueThrowsForUnsupportedInstance), "has number",
				"Run Number", "\t5")).ParseMembersAndMethods(new MethodExpressionParser());
		var instance = new ValueInstance(customType,
			new Dictionary<string, ValueInstance> { { "number", new ValueInstance(numberType, 1d) } });
		var charType = TestPackage.Instance.GetType(Type.Character);
		Assert.Throws<ValueInstance.IteratorNotSupported>(() =>
			_ = instance.GetIteratorValue(charType, 0));
	}

	[Test]
	public void EqualsReturnsTrueWhenTypeIdHasNumberMemberMatchingPrimitive()
	{
		using var t = new Type(TestPackage.Instance,
			new TypeLines("TypeIdNumberMemberMatchesPrimitive",
				"has number", "Run Boolean",
				"\tnumber is 1")).ParseMembersAndMethods(new MethodExpressionParser());
		var typeInstance = new ValueInstance(t,
			new Dictionary<string, ValueInstance> { { "number", new ValueInstance(numberType, 42d) } });
		Assert.That(typeInstance, Is.EqualTo(new ValueInstance(numberType, 42d)));
	}

	[Test]
	public void EqualsReturnsTrueWhenPrimitiveMatchesTypeIdWithNumberMember()
	{
		using var t = new Type(TestPackage.Instance,
			new TypeLines("PrimitiveMatchesTypeIdNumberMember",
				"has number", "Run Boolean",
				"\tnumber is 1")).ParseMembersAndMethods(new MethodExpressionParser());
		var typeInstance = new ValueInstance(t,
			new Dictionary<string, ValueInstance> { { "number", new ValueInstance(numberType, 42d) } });
		Assert.That(new ValueInstance(numberType, 42d), Is.EqualTo(typeInstance));
	}
}