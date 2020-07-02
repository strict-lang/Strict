using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class PackageTests
	{
		[Test]
		public void LoadingUnknownPackageIsNotAllowedYet() =>
			Assert.Throws<Package.OnlyStrictPackageIsAllowed>(() =>
				Package.FromDisk(nameof(LoadingUnknownPackageIsNotAllowedYet)));

		[Test]
		public void NoneAndBooleanAreAlwaysKnown()
		{
			var emptyPackage = new Package(nameof(NoneAndBooleanAreAlwaysKnown));
			Assert.That(emptyPackage.FindType(Base.None), Is.Not.Null);
			Assert.That(emptyPackage.FindType(Base.Boolean), Is.Not.Null);
			Assert.That(emptyPackage.FindType(Base.Number), Is.Null);
		}

		//ncrunch: no coverage start
		[Test, Ignore("Parsing not done for method-less new base code")]
		public void LoadStrictBaseTypes()
		{
			//we first need to load all type names in a folder and allow all classes to use each others type, then fill them in!
			var package = Package.FromDisk(nameof(Strict));
			Assert.That(package.FindDirectType(Base.Any), Is.Not.Null);
			Assert.That(package.FindDirectType(Base.Number), Is.Not.Null);
			Assert.That(package.FindDirectType(Base.Type), Is.Not.Null);
		}
	}
}