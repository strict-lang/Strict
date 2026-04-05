using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class ValueInstanceTests
{
	[SetUp]
	public void CreateNumber() => numberType = TestPackage.Instance.GetType(Type.Number);

	private Type numberType = null!;

	[Test]
	public void ToStringShowsTypeAndValue() =>
		Assert.That(new ValueInstance(numberType, 42).ToString(), Is.EqualTo("Number: 42"));

	[Test]
	public void ValueInstancePerformanceLoggingWritesConstructorAndCodeStringLogs()
	{
		using var consoleWriter = new StringWriter();
		var rememberLogging = Environment.GetEnvironmentVariable("STRICT_PERFORMANCE_LOGGING");
   var rememberIsEnabled = PerformanceLog.IsEnabled;
		var logWriterField = typeof(PerformanceLog).GetField("logWriter",
			System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
		var rememberLogWriter = logWriterField.GetValue(null);
		logWriterField.SetValue(null, consoleWriter);
		Environment.SetEnvironmentVariable("STRICT_PERFORMANCE_LOGGING", "1");
   PerformanceLog.IsEnabled = true;
		try
		{
			var noneType = TestPackage.Instance.GetType(Type.None);
			var booleanType = TestPackage.Instance.GetType(Type.Boolean);
			var listType = TestPackage.Instance.GetListImplementationType(numberType);
			var dictionaryType = TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType);
     using var customType = new Type(TestPackage.Instance,
				new TypeLines("ValueInstanceLoggingSample",
					"has number", "Run Number", "\t5")).ParseMembersAndMethods(new MethodExpressionParser());
			_ = new ValueInstance(noneType).ToExpressionCodeString();
			_ = new ValueInstance(booleanType, true).ToExpressionCodeString();
			_ = new ValueInstance(numberType, 42).ToExpressionCodeString();
			_ = new ValueInstance("hello").ToExpressionCodeString(true);
			_ = new ValueInstance(dictionaryType, new Dictionary<ValueInstance, ValueInstance>
			{
				{ new ValueInstance(numberType, 1), new ValueInstance(numberType, 2) }
			}).ToExpressionCodeString();
			_ = new ValueInstance(listType, [new ValueInstance(numberType, 1)]).ToExpressionCodeString();
      var typeInstance = new ValueInstance(customType, [new ValueInstance(numberType, 7)]);
			_ = GetExpressionCodeStringViaHelperEntry(typeInstance);
			_ = new ValueInstance(typeInstance, customType).ToExpressionCodeString();
		}
		finally
		{
     logWriterField.SetValue(null, rememberLogWriter);
			PerformanceLog.IsEnabled = rememberIsEnabled;
			Environment.SetEnvironmentVariable("STRICT_PERFORMANCE_LOGGING", rememberLogging);
		}
		var output = consoleWriter.ToString();
    Assert.That(output, Does.Contain("ValueInstance.ctor(Type=None)"));
   Assert.That(output, Does.Contain("ValueInstance.ctor(Type=TestPackage/Boolean"));
		Assert.That(output, Does.Contain("number=True"));
    Assert.That(output, Does.Contain("ValueInstance.ctor(Type=TestPackage/Number"));
		Assert.That(output, Does.Contain("number=42"));
		Assert.That(output, Does.Contain("ValueInstance.ctor(text=hello)"));
    Assert.That(output, Does.Contain("ValueInstance.ctor(Type=TestPackage/Dictionary("));
		Assert.That(output, Does.Contain("ValueInstance.ctor(Type=TestPackage/List("));
		Assert.That(output, Does.Contain("ValueInstance.ctor(existingInstance="));
		Assert.That(output, Does.Contain("ValueInstance.ToExpressionCodeString"));
   Assert.That(output, Does.Contain("generated=(7)"));
		Assert.That(output, Does.Contain("callers="));
   Assert.That(output, Does.Contain(nameof(GetExpressionCodeStringViaHelperEntry)));
   Assert.That(output, Does.Contain(nameof(GetExpressionCodeStringViaHelperRoot)));
		Assert.That(output, Does.Contain(nameof(GetExpressionCodeStringViaHelpers)));
	}

	private static string GetExpressionCodeStringViaHelperEntry(ValueInstance instance) =>
		GetExpressionCodeStringViaHelperRoot(instance);

	private static string GetExpressionCodeStringViaHelperRoot(ValueInstance instance) =>
		GetExpressionCodeStringViaHelpers(instance);

	private static string GetExpressionCodeStringViaHelpers(ValueInstance instance) =>
		GetExpressionCodeStringViaHelper(instance);

	private static string GetExpressionCodeStringViaHelper(ValueInstance instance) =>
		instance.ToExpressionCodeString();

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
	public void ValueInstanceCanWrapExistingListWithoutCopying()
	{
		var listType = TestPackage.Instance.GetListImplementationType(numberType);
		List<ValueInstance> items = [new(numberType, 1), new(numberType, 2)];
    var constructor = typeof(ValueInstance).GetConstructor([typeof(Type), typeof(List<ValueInstance>), typeof(bool)]);
		Assert.That(constructor, Is.Not.Null);
   var instance = (ValueInstance)constructor!.Invoke([listType, items, true]);
		Assert.That(instance.List.Items, Is.SameAs(items));
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
		var flatNumbersField = typeof(ValueListInstance).GetField("flatNumbers",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
   var itemsField = typeof(ValueListInstance).GetField("items",
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
	public void ValueListInstanceStoresTypeAndItems()
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
      new TypeLines("CompactRgbaValue",
				"has Red Number",
				"has Green Number",
				"has Blue Number",
				"has Alpha = 1",
				"Run Number",
				"\tRed")).ParseMembersAndMethods(new MethodExpressionParser());
		var instance = new ValueInstance(rgbaType, [
			new ValueInstance(numberType, 0.5),
			new ValueInstance(numberType, 0.75),
			new ValueInstance(numberType, 0.5),
			new ValueInstance(numberType, 1)
		]);
   var valueField = typeof(ValueInstance).GetField("value",
			System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
		Assert.That(valueField.GetValue(instance)!.GetType().Name, Is.Not.EqualTo("ValueTypeInstance"));
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
}