using NUnit.Framework;

namespace Strict.Language.Tests;

public class TypeLinesTests
{
	[Test]
	public void ListMembersShouldBeExtractedCorrectly()
	{
		var type = new TypeLines(Base.Text, "has characters");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(2));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Base.List));
		Assert.That(type.DependentTypes[1], Is.EqualTo(Base.Character));
	}

	[Test]
	public void OutputShouldBeUppercase()
	{
		var type = new TypeLines(Base.Log, "has output");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(1));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Base.Output));
	}

	[Test]
	public void RangeHasIteratorAndNumber()
	{
		var type = new TypeLines(Base.Range, "has iterator", "has Start Number", "has End Number");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(2));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Base.Iterator));
		Assert.That(type.DependentTypes[1], Is.EqualTo(Base.Number));
	}

	[Test]
	public void MethodReturnTypeShouldBeExtractedIntoDependentTypes()
	{
		var type = new TypeLines(Base.Directory,
			"GetFile Text",
			"GetFiles Texts",
			"GetDirectories Texts");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(2), string.Join(",", type.DependentTypes));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Base.Text));
		Assert.That(type.DependentTypes[1], Is.EqualTo(Base.List));
	}
}