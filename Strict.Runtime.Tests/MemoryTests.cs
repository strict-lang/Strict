namespace Strict.Runtime.Tests;

public sealed class MemoryTests
{
	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);

	[Test]
	public void AddToCollectionVariableDoesNothingWhenValueIsNotAList()
	{
		var memory = new Memory { Variables = { ["count"] = new ValueInstance(NumberType, 5.0) } };
		memory.AddToCollectionVariable("count", new ValueInstance(NumberType, 1.0));
		Assert.That(memory.Variables["count"].Number, Is.EqualTo(5));
	}

	[Test]
	public void AddToCollectionVariableAddsElementToNonEmptyList()
	{
		var memory = new Memory();
		var listType = TestPackage.Instance.GetListImplementationType(NumberType);
		memory.Variables["items"] = new ValueInstance(listType, [new(NumberType, 1.0)]);
		memory.AddToCollectionVariable("items", new ValueInstance(NumberType, 2.0));
		Assert.That(memory.Variables["items"].List.Items.Count, Is.EqualTo(2));
	}
}
