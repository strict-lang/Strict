using NUnit.Framework;

namespace Strict.Language.Tests;

public class TypeLinesTests
{
	[Test]
	public void ListMembersShouldBeExtractedCorrectly()
	{
		var type = new TypeLines(Base.Text, "has characters");
		Assert.That(type.MemberTypes.Count, Is.EqualTo(2));
		Assert.That(type.MemberTypes[0], Is.EqualTo(Base.List));
		Assert.That(type.MemberTypes[1], Is.EqualTo(Base.Character));
	}

	[Test]
	public void OutputShouldBeUppercase()
	{
		var type = new TypeLines(Base.Log, "has output");
		Assert.That(type.MemberTypes.Count, Is.EqualTo(1));
		Assert.That(type.MemberTypes[0], Is.EqualTo(Base.Output));
	}

	[Test]
	public void RangeHasIteratorAndNumber()
	{
		var type = new TypeLines(Base.Range, "has iterator", "has Start Number", "has End Number");
		Assert.That(type.MemberTypes.Count, Is.EqualTo(2));
		Assert.That(type.MemberTypes[0], Is.EqualTo(Base.Iterator));
		Assert.That(type.MemberTypes[1], Is.EqualTo(Base.Number));
	}
}