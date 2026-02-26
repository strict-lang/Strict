using Strict.Expressions;
using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class ValueInstanceTests
{
	[SetUp]
	public void CreateNumber() => numberType = TestPackage.Instance.GetType(Base.Number);

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
		var typeInstance = new ValueTypeInstance(t,
			new Dictionary<string, ValueInstance> { { "number", new ValueInstance(numberType, 7d) } });
		var vi = new ValueInstance(typeInstance);
		Assert.That(vi.ToString(), Does.Contain(nameof(ValueTypeInstanceStoresMembers)));
	}
}