using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class PackageTests
	{
		[Test]
		public void NoneAndBooleanAreAlwaysKnown()
		{
			var emptyPackage = new Package(nameof(NoneAndBooleanAreAlwaysKnown));
			Assert.That(emptyPackage.FindType(Base.None), Is.Not.Null);
			Assert.That(emptyPackage.FindType(Base.Boolean), Is.Not.Null);
			Assert.That(emptyPackage.FindType(Base.Number), Is.Null);
		}
	}
}