namespace Strict.Runtime.Tests;

public sealed class MemoryTests
{
	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);

	[Test]
	public void AddToCollectionVariableDoesNothingWhenValueIsNotAList()
	{
		var memory = new Memory();
		memory.Variables["count"] = new ValueInstance(NumberType, 5.0);
		memory.AddToCollectionVariable("count", new ValueInstance(NumberType, 1.0));
		Assert.That(memory.Variables["count"].Number, Is.EqualTo(5));
	}

	[Test]
	public void AddToCollectionVariableThrowsWhenEmptyListHasNonGenericType()
	{
		var memory = new Memory();
		var rawListType = TestPackage.Instance.GetType(Type.List);
		memory.Variables["items"] = new ValueInstance(rawListType, new List<ValueInstance>());
		Assert.That(() => memory.AddToCollectionVariable("items", new ValueInstance(NumberType, 1.0)),
			Throws.InstanceOf<InvalidOperationException>());
	}

	[Test]
	public void AddToCollectionVariableAddsElementToNonEmptyList()
	{
		var memory = new Memory();
		var listType = TestPackage.Instance.GetListImplementationType(NumberType);
		memory.Variables["items"] = new ValueInstance(listType,
			new List<ValueInstance> { new(NumberType, 1.0) });
		memory.AddToCollectionVariable("items", new ValueInstance(NumberType, 2.0));
		Assert.That(memory.Variables["items"].List.Items.Count, Is.EqualTo(2));
	}
}
