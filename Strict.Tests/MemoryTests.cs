using Strict.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Tests;

public sealed class MemoryTests
{
	[Test]
	public void AddToCollectionVariableThrowsWhenValueIsNotAList()
	{
		var memory = new Memory { Variables = { ["count"] = new ValueInstance(NumberType, 5) } };
		Assert.That(() => memory.AddToCollection("count", new ValueInstance(NumberType, 1)),
			Throws.InvalidOperationException);
	}

	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);

	[Test]
	public void AddToCollectionVariableAddsElementToNonEmptyList()
	{
		var memory = new Memory();
		var listType = TestPackage.Instance.GetListImplementationType(NumberType);
		memory.Variables["items"] = new ValueInstance(listType, [new(NumberType, 1)]);
		memory.AddToCollection("items", new ValueInstance(NumberType, 2));
		Assert.That(memory.Variables["items"].List.Items.Count, Is.EqualTo(2));
	}
}