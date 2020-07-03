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

		[Test]
		public void LoadStrictBaseTypes()
		{
			var package = Package.FromDisk(nameof(Strict)).GetSubPackage(nameof(Base));
			Assert.That(package.FindDirectType(Base.Any), Is.Not.Null);
			Assert.That(package.FindDirectType(Base.Number), Is.Not.Null);
			Assert.That(package.FindDirectType(Base.App), Is.Not.Null);
		}
	}
}