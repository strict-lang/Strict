using System.Threading.Tasks;
using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class PackageTests
	{
		[SetUp]
		public void CreateContexts()
		{
			mainPackage = new Package(nameof(TestPackage)); 
			mainType = new Type(mainPackage, "Yolo", "dummy");
			subPackage = new Package(mainPackage, nameof(PackageTests));
			privateSubType = new Type(subPackage, "secret", "dummy");
			publicSubType = new Type(subPackage, "FindMe", "dummy");
		}

		private Package mainPackage;
		private Type mainType;
		private Package subPackage;
		private Type privateSubType;
		private Type publicSubType;
		
		[Test]
		public void NoneAndBooleanAreAlwaysKnown()
		{
			var emptyPackage = new Package(nameof(NoneAndBooleanAreAlwaysKnown));
			Assert.That(emptyPackage.FindType(Base.None, emptyPackage), Is.Not.Null);
			Assert.That(emptyPackage.FindType(Base.Boolean, emptyPackage), Is.Not.Null);
			Assert.That(emptyPackage.FindType(nameof(NoneAndBooleanAreAlwaysKnown), emptyPackage),
				Is.Null);
		}

		[Test]
		public void GetFullNames()
		{
			Assert.That(mainPackage.ToString(), Is.EqualTo(nameof(TestPackage)));
			Assert.That(mainType.ToString(), Is.EqualTo(nameof(TestPackage) + "." + mainType.Name));
			Assert.That(subPackage.ToString(),
				Is.EqualTo(nameof(TestPackage) + "." + nameof(PackageTests)));
			Assert.That(privateSubType.ToString(),
				Is.EqualTo(nameof(TestPackage) + "." + nameof(PackageTests) + "." +
					privateSubType.Name));
			Assert.That(publicSubType.ToString(),
				Is.EqualTo(
					nameof(TestPackage) + "." + nameof(PackageTests) + "." + publicSubType.Name));
		}

		[Test]
		public void PrivateTypesCanOnlyBeFoundInPackageTheyAreIn()
		{
			Assert.That(mainType.GetType(publicSubType.Name), Is.EqualTo(publicSubType));
			Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
				mainPackage.GetType(privateSubType.ToString()));
			Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
				mainPackage.GetType(nameof(TestPackage) + "." + nameof(PackageTests) + "." +
					privateSubType.Name));
		}

		[Test]
		public void FindSubTypeBothWays()
		{
			Assert.That(mainType.GetType(publicSubType.ToString()), Is.EqualTo(publicSubType));
			Assert.That(publicSubType.GetType(mainType.ToString()), Is.EqualTo(mainType));
		}

		[Test]
		public void FindingFullTypeRequiresFullName() =>
			Assert.Throws<Package.FullNameMustContainPackageAndTypeNames>(() =>
				mainPackage.FindFullType(publicSubType.Name));

		[Test]
		public void ContextNameMustNotContainSpecialCharactersOrNumbers()
		{
			Assert.Throws<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>(() =>
				new Type(mainPackage, "MyClass123", ""));
			Assert.Throws<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>(() =>
				new Package(mainPackage, "$%"));
		}
		
		[Test]
		public async Task LoadTypesFromOtherPackage()
		{
			var strictPackage = await new Repositories().LoadFromUrl(Repositories.StrictUrl);
			Assert.That(mainPackage.GetType(Base.App),
				Is.EqualTo(strictPackage.GetType(Base.App)).And.
					EqualTo(subPackage.GetType(Base.App)));
			Assert.That(mainPackage.GetType(Base.Character),
				Is.Not.EqualTo(mainPackage.GetType(Base.App)));
		}

		/// <summary>
		/// Can be used to profile and optimize the GetType performance by doing it many times
		/// </summary>
		[Test]
		public void LoadingTypesOverAndOverWillAlwaysQuicklyReturnTheSame()
		{
			var otherMainPackage =
				new Package(nameof(LoadingTypesOverAndOverWillAlwaysQuicklyReturnTheSame));
			for (int index = 0; index < 1000000; index++)
				if (otherMainPackage.FindType(mainType.Name, otherMainPackage) != mainType)
					throw new AssertionException("FindType failed"); //ncrunch: no coverage
		}
	}
}