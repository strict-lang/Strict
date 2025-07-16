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
	public void TextWriterShouldBeUppercase()
	{
		var type = new TypeLines(Base.Logger, "has textWriter");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(1));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Base.TextWriter));
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
		var typeLines = new TypeLines(Base.Directory,
			"GetFile Text",
			"GetFiles Texts",
			"GetDirectories Texts");
		Assert.That(typeLines.DependentTypes.Count, Is.EqualTo(2),
			string.Join(",", typeLines.DependentTypes));
		Assert.That(typeLines.DependentTypes[0], Is.EqualTo(Base.Text));
		Assert.That(typeLines.DependentTypes[1], Is.EqualTo(Base.List));
	}
}