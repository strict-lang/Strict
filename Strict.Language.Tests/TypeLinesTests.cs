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
}