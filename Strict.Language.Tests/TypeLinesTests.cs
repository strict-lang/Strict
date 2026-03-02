namespace Strict.Language.Tests;

public class TypeLinesTests
{
	[Test]
	public void ListMembersShouldBeExtractedCorrectly()
	{
		var type = new TypeLines(Type.Text, "has characters");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(2));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Type.List));
		Assert.That(type.DependentTypes[1], Is.EqualTo(Type.Character));
	}

	[Test]
	public void TextWriterShouldBeUppercase()
	{
		var type = new TypeLines(Type.Logger, "has textWriter");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(1));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Type.TextWriter));
	}

	[Test]
	public void RangeHasIteratorAndNumber()
	{
		var type = new TypeLines(Type.Range, "has iterator", "has Start Number", "has End Number");
		Assert.That(type.DependentTypes.Count, Is.EqualTo(2));
		Assert.That(type.DependentTypes[0], Is.EqualTo(Type.Iterator));
		Assert.That(type.DependentTypes[1], Is.EqualTo(Type.Number));
	}

	[Test]
	public void MethodReturnTypeShouldBeExtractedIntoDependentTypes()
	{
		var typeLines = new TypeLines(Type.Directory,
			"GetFile Text",
			"GetFiles Texts",
			"GetDirectories Texts");
		Assert.That(typeLines.DependentTypes.Count, Is.EqualTo(2),
			string.Join(",", typeLines.DependentTypes));
		Assert.That(typeLines.DependentTypes[0], Is.EqualTo(Type.Text));
		Assert.That(typeLines.DependentTypes[1], Is.EqualTo(Type.List));
	}
}