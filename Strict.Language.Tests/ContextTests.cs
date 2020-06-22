using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class ContextTests
	{
		[SetUp]
		public void CreateContexts()
		{
			mainPackage = new Package(nameof(TestPackage)); 
			mainType = new Type(mainPackage, "Yolo", "method dummy");
			subPackage = new Package(mainPackage, nameof(ContextTests));
			subType = new Type(subPackage, "secret", "method dummy");
		}

		private Package mainPackage;
		private Type mainType;
		private Package subPackage;
		private Type subType;

		[Test]
		public void GetFullNames()
		{
			Assert.That(mainPackage.FullName, Is.EqualTo(nameof(TestPackage)));
			Assert.That(mainType.FullName, Is.EqualTo(nameof(TestPackage) + "." + mainType.Name));
			Assert.That(subPackage.FullName,
				Is.EqualTo(nameof(TestPackage) + "." + nameof(ContextTests)));
			Assert.That(subType.FullName,
				Is.EqualTo(nameof(TestPackage) + "." + nameof(ContextTests) + "." + subType.Name));
		}
	}
}