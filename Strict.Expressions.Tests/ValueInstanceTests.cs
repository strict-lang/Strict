using Strict.Language.Tests;
using System.Runtime.InteropServices;

namespace Strict.Expressions.Tests;

public sealed class ValueInstanceTests
{
	[SetUp]
	public void CreateNumber() => numberType = TestPackage.Instance.GetType(Type.Number);

	private Type numberType = null!;

	[Test]
	public void ValueInstanceIsAlwaysJustTwoValuesAsStruct() =>
		Assert.That(Marshal.SizeOf(typeof(ValueInstance)), Is.EqualTo(2 * sizeof(double)));

	[Test]
	public void ToStringShowsTypeAndValue() =>
		Assert.That(new ValueInstance(numberType, 42).ToString(), Is.EqualTo("Number: 42"));

	[Test]
	public void CompareTwoNumbers() =>
		Assert.That(new ValueInstance(numberType, 42),
			Is.EqualTo(new ValueInstance(numberType, 42)));

	[Test]
	public void CompareNumberToText() =>
		Assert.That(new ValueInstance(numberType, 5),
			Is.Not.EqualTo(new ValueInstance("5")));

	[Test]
	public void CompareLists()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var list = new ValueInstance(listType, [
			new ValueInstance(numberType, 1),
			new ValueInstance(numberType, 2),
			new ValueInstance(numberType, 3)
		]);
		Assert.That(list, Is.EqualTo(new ValueInstance(listType, [
			new ValueInstance(numberType, 1),
			new ValueInstance(numberType, 2),
			new ValueInstance(numberType, 3)
		])));
		Assert.That(list, Is.Not.EqualTo(new ValueInstance(listType, [
			new ValueInstance(numberType, 1),
			new ValueInstance(numberType, 2),
			new ValueInstance(numberType, 1)
		])));
	}

	[Test]
	public void ListWithValueInstancesWorks()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		Assert.That(new ValueInstance(listType, [new Number(numberType, 1).Data]), Is.Not.Null);
	}

	[Test]
	public void GenericListTypeAcceptsValueInstances()
	{
		var listType = TestPackage.Instance.GetType("List(key Generic, mappedValue Generic)");
		Assert.That(new ValueInstance(listType, Array.Empty<ValueInstance>()), Is.Not.Null);
	}

	[Test]
	public void ValueInstanceDoesNotExposeListReuseConstructor()
	{
		var constructor = typeof(ValueInstance).GetConstructor([
			typeof(Type), typeof(List<ValueInstance>), typeof(bool)
		]);
		Assert.That(constructor, Is.Null);
	}

	[Test]
	public void ValueInstanceCreatesEmptyListWithRequestedCapacity()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var instance = ValueInstance.CreateListWithCapacity(listType, 12);
		Assert.That(instance.List.Items.Count, Is.EqualTo(0));
		Assert.That(instance.List.Items.Capacity, Is.GreaterThanOrEqualTo(12));
	}

	[Test]
	public void NumericDataTypeListsUseFlatNumberBacking()
	{
		using var pointType = new Type(TestPackage.Instance,
				new TypeLines("Point2", "has xValue Number", "has yValue Number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var listType = TestPackage.Instance.GetListImplementationType(pointType);
		var point1 = new ValueInstance(pointType,
			[new ValueInstance(numberType, 1), new ValueInstance(numberType, 2)]);
		var point2 = new ValueInstance(pointType,
			[new ValueInstance(numberType, 3), new ValueInstance(numberType, 4)]);
		var list = new ValueInstance(listType, [point1, point2]);
		var flatNumbersField = typeof(ValueArrayInstance).GetField("flatNumbers",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
		var itemsField = typeof(ValueArrayInstance).GetField("items",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
		Assert.That(flatNumbersField, Is.Not.Null);
		Assert.That(itemsField, Is.Not.Null);
		Assert.That((float[]?)flatNumbersField!.GetValue(list.List), Is.Not.Null);
		Assert.That(itemsField!.GetValue(list.List), Is.Null);
		Assert.That(list.List[1].TryGetValueTypeInstance()!["yValue"].Number, Is.EqualTo(4));
	}

	[Test]
	public void CompareDictionaries()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var k1 = new ValueInstance(numberType, 1);
		var k2 = new ValueInstance(numberType, 2);
		var v2 = new ValueInstance(numberType, 2);
		var v3 = new ValueInstance(numberType, 3);
		var d1 = new Dictionary<ValueInstance, ValueInstance> { { k1, v2 } };
		var d2 = new Dictionary<ValueInstance, ValueInstance>
		{
			{ new ValueInstance(numberType, 1), new ValueInstance(numberType, 2) }
		};
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
		Assert.That(new ValueInstance(t, 42), Is.EqualTo(new ValueInstance(numberType, 42)));
	}

	[Test]
	public void ValueArrayInstanceStoresTypeAndItems()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		ValueInstance[] items = [new(numberType, 1), new(numberType, 2)];
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
		var instance = new ValueInstance(t, [new ValueInstance(numberType, 7)]);
		Assert.That(instance.ToString(), Does.Contain(nameof(ValueTypeInstanceStoresMembers)));
	}

	[Test]
	public void ValueInstanceWithRgbaMembersUsesCompactRepresentation()
	{
		using var rgbaType = new Type(TestPackage.Instance,
				new TypeLines("CompactRgbaValue", "has Red Number", "has Green Number", "has Blue Number",
					"has Alpha = 1", "Run Number", "\tRed")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var instance = new ValueInstance(rgbaType, [
			new ValueInstance(numberType, 0.5),
			new ValueInstance(numberType, 0.75),
			new ValueInstance(numberType, 0.5),
			new ValueInstance(numberType, 1)
		]);
		var valueField = typeof(ValueInstance).GetField("value",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
		Assert.That(valueField.GetValue(instance)!.GetType().Name,
			Is.Not.EqualTo("ValueTypeInstance"));
		Assert.That(instance.ToExpressionCodeString(), Is.EqualTo("(0.5, 0.75, 0.5)"));
	}

	[Test]
	public void ThrowsInvalidTypeValueWhenUsingReservedNumberForText() =>
		Assert.Throws<ValueInstance.InvalidTypeValue>(() =>
			_ = new ValueInstance(numberType, -7.90897526e307));

	[Test]
	public void ThrowsWhenCreatingTypeInstanceForNumberType() =>
		Assert.Throws<ValueInstance.ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes>(() =>
			_ = new ValueInstance(numberType, Array.Empty<ValueInstance>()));

	[Test]
	public void CopyConstructorWithDictionaryCreatesNewReturnType()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
		var mutableDictType =
			TestPackage.Instance.GetType(Type.Mutable).GetGenericImplementation(dictType);
		var original = new ValueInstance(dictType,
			new Dictionary<ValueInstance, ValueInstance>
			{
				{ new ValueInstance(numberType, 1), new ValueInstance(numberType, 2) }
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
		var original = new ValueInstance(originalType, [new ValueInstance(numberType, 5)]);
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
		Assert.That(listType.IsList, Is.True);
		var mutableListType =
			TestPackage.Instance.GetType(Type.Mutable).GetGenericImplementation(listType);
		Assert.That(mutableListType.IsMutable, Is.True);
		Assert.That(mutableListType.IsList, Is.True);
		var mutableInstance = new ValueInstance(mutableListType, [new ValueInstance(numberType, 1)]);
		Assert.That(mutableInstance.IsMutable, Is.True);
		Assert.That(mutableInstance.IsList, Is.True);
		var result = mutableInstance.ApplyMethodReturnTypeMutable(listType);
		Assert.That(result.IsMutable, Is.False);
		Assert.That(result.IsList, Is.True);
	}

	[Test]
	public void GetTypeExceptTextReturnsListReturnTypeForListInstance()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var instance = new ValueInstance(listType, Array.Empty<ValueInstance>());
		Assert.That(instance.GetType(), Is.EqualTo(listType));
	}

	[Test]
	public void GetIteratorLengthThrowsForDictionaryInstance() =>
		Assert.Throws<ValueInstance.IteratorNotSupported>(() =>
			new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<ValueInstance, ValueInstance>()).GetIteratorLength());

	[Test]
	public void GetIteratorLengthForTypeIdWithKeysAndValuesMember()
	{
		using var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(GetIteratorLengthForTypeIdWithKeysAndValuesMember),
				"has number", "has keysAndValues Numbers",
				"Run Number", "\t5")).ParseMembersAndMethods(new MethodExpressionParser());
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var listInstance = new ValueInstance(listType,
		[
			new ValueInstance(numberType, 1), new ValueInstance(numberType, 2),
			new ValueInstance(numberType, 3)
		]);
		var instance =
			new ValueInstance(customType, [new ValueInstance(numberType, 1), listInstance]);
		Assert.That(instance.GetIteratorLength(), Is.EqualTo(3));
	}

	[Test]
	public void GetIteratorValueForTypeIdWithElementsMember()
	{
		using var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(GetIteratorValueForTypeIdWithElementsMember),
				"has number", "has elements Numbers",
				"Run Number", "\t5")).ParseMembersAndMethods(new MethodExpressionParser());
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		var item1 = new ValueInstance(numberType, 10);
		var item2 = new ValueInstance(numberType, 20);
		var listInstance = new ValueInstance(listType, [item1, item2]);
		var instance =
			new ValueInstance(customType, [new ValueInstance(numberType, 1), listInstance]);
		var charType = TestPackage.Instance.GetType(Type.Character);
		Assert.That(instance.GetIteratorValue(charType, 1), Is.EqualTo(item2));
	}

	[Test]
	public void GetIteratorValueThrowsForUnsupportedInstance()
	{
		using var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(GetIteratorValueThrowsForUnsupportedInstance), "has number",
				"Run Number", "\t5")).ParseMembersAndMethods(new MethodExpressionParser());
		var instance = new ValueInstance(customType, [new ValueInstance(numberType, 1)]);
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
		var typeInstance = new ValueInstance(t, [new ValueInstance(numberType, 42)]);
		Assert.That(typeInstance, Is.EqualTo(new ValueInstance(numberType, 42)));
	}

	[Test]
	public void EqualsReturnsTrueWhenPrimitiveMatchesTypeIdWithNumberMember()
	{
		using var t = new Type(TestPackage.Instance,
			new TypeLines("PrimitiveMatchesTypeIdNumberMember",
				"has number", "Run Boolean",
				"\tnumber is 1")).ParseMembersAndMethods(new MethodExpressionParser());
		var typeInstance = new ValueInstance(t, [new ValueInstance(numberType, 42)]);
		Assert.That(new ValueInstance(numberType, 42), Is.EqualTo(typeInstance));
	}

	[Test]
	public void SizeTypeUsesFlatFloatArrayBacking()
	{
		using var sizeType = new Type(TestPackage.Instance,
				new TypeLines("FlatSize", "has Width Number", "has Height Number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var size = ValueInstance.CreateFlatNumericType(sizeType, [128f, 72f]);
		var valueField = typeof(ValueInstance).GetField("value",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
		Assert.That(valueField.GetValue(size), Is.InstanceOf<ValueArrayInstance>());
		Assert.That(size.GetType(), Is.EqualTo(sizeType));
		Assert.That(size.TryGetFlatNumericMember("Width", out var width), Is.True);
		Assert.That(width.Number, Is.EqualTo(128));
		Assert.That(size.TryGetFlatNumericMember("Height", out var height), Is.True);
		Assert.That(height.Number, Is.EqualTo(72));
	}

	[Test]
	public void ColorTypeUsesFlatFloatArrayBacking()
	{
		using var colorType = new Type(TestPackage.Instance,
				new TypeLines("FlatColor", "has Hue Number", "has Saturation Number",
					"has Lightness Number", "has Opacity Number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var color = ValueInstance.CreateFlatNumericType(colorType,
			[0.5f, 0.75f, 0.25f, 1f]);
		var valueField = typeof(ValueInstance).GetField("value",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
		Assert.That(valueField.GetValue(color), Is.InstanceOf<ValueArrayInstance>());
		Assert.That(color.TryGetFlatNumericMember("Hue", out var hue), Is.True);
		Assert.That(hue.Number, Is.EqualTo(0.5).Within(0.001));
		Assert.That(color.TryGetFlatNumericMember("Saturation", out var saturation), Is.True);
		Assert.That(saturation.Number, Is.EqualTo(0.75).Within(0.001));
		Assert.That(color.TryGetFlatNumericMember("Lightness", out var lightness), Is.True);
		Assert.That(lightness.Number, Is.EqualTo(0.25).Within(0.001));
		Assert.That(color.TryGetFlatNumericMember("Opacity", out var opacity), Is.True);
		Assert.That(opacity.Number, Is.EqualTo(1).Within(0.001));
	}

	[Test]
	public void ListOfColorsUsesSharedFlatBackingArray()
	{
		using var colorType = new Type(TestPackage.Instance,
				new TypeLines("FlatColor2", "has Hue Number", "has Saturation Number",
					"has Lightness Number", "has Opacity Number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var listType = TestPackage.Instance.GetListImplementationType(colorType);
		var flatNumbers = new float[40];
		for (var colorIndex = 0; colorIndex < 10; colorIndex++)
		{
			flatNumbers[colorIndex * 4] = colorIndex * 0.1f;
			flatNumbers[colorIndex * 4 + 1] = colorIndex * 0.05f;
			flatNumbers[colorIndex * 4 + 2] = colorIndex * 0.02f;
			flatNumbers[colorIndex * 4 + 3] = 1f;
		}
		var list = ValueInstance.CreateFlatNumericList(listType, colorType, flatNumbers, 4);
		var flatNumbersField = typeof(ValueArrayInstance).GetField("flatNumbers",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
		var backingArray = (float[]?)flatNumbersField.GetValue(list.List);
		Assert.That(backingArray, Is.Not.Null);
		Assert.That(backingArray!.Length, Is.EqualTo(40));
		var element5 = list.List[5];
		Assert.That(element5.IsFlatNumeric, Is.True);
		Assert.That(element5.TryGetFlatNumericMember("Hue", out var hue5), Is.True);
		Assert.That(hue5.Number, Is.EqualTo(0.5).Within(0.01));
		Assert.That(element5.TryGetFlatNumericMember("Opacity", out var opacity5), Is.True);
		Assert.That(opacity5.Number, Is.EqualTo(1).Within(0.001));
		var valueField = typeof(ValueInstance).GetField("value",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
		var elementBacking = (ValueArrayInstance)valueField.GetValue(element5)!;
		var elementBackingNumbers = typeof(ValueArrayInstance).GetField("flatNumbers",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
		Assert.That(elementBackingNumbers.GetValue(elementBacking),
			Is.SameAs(backingArray), "Slice should reference same backing array");
	}
}