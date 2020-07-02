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
			privateSubType = new Type(subPackage, "secret", "method dummy");
			publicSubType = new Type(subPackage, "FindMe", "method dummy");
		}

		private Package mainPackage;
		private Type mainType;
		private Package subPackage;
		private Type privateSubType;
		private Type publicSubType;

		[Test]
		public void GetFullNames()
		{
			Assert.That(mainPackage.ToString(), Is.EqualTo(nameof(TestPackage)));
			Assert.That(mainType.ToString(), Is.EqualTo(nameof(TestPackage) + "." + mainType.Name));
			Assert.That(subPackage.ToString(),
				Is.EqualTo(nameof(TestPackage) + "." + nameof(ContextTests)));
			Assert.That(privateSubType.ToString(),
				Is.EqualTo(nameof(TestPackage) + "." + nameof(ContextTests) + "." +
					privateSubType.Name));
			Assert.That(publicSubType.ToString(),
				Is.EqualTo(
					nameof(TestPackage) + "." + nameof(ContextTests) + "." + publicSubType.Name));
		}

		[Test]
		public void PrivateTypesCanOnlyBeFoundInPackageTheyAreIn()
		{
			Assert.That(mainType.GetType(publicSubType.Name), Is.EqualTo(publicSubType));
			Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
				mainPackage.GetType(privateSubType.Name));
			Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
				mainPackage.GetType(nameof(TestPackage) + "." + nameof(ContextTests) + "." +
					privateSubType.Name));
		}

		[Test]
		public void FindSubType() =>
			Assert.That(mainType.GetType(publicSubType.ToString()), Is.EqualTo(publicSubType));

		[Test]
		public void ContextNameMustNotContainSpecialCharactersOrNumbers()
		{
			Assert.Throws<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>(() =>
				new Type(mainPackage, "MyClass123", ""));
			Assert.Throws<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>(() =>
				new Package(mainPackage, "$%"));
		}
	}
}