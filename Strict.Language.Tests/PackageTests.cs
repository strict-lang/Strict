using System;
using System.Threading.Tasks;
using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class PackageTests
	{
		[Test]
		public void InvalidUrlWontWork() =>
			Assert.ThrowsAsync<UriFormatException>(async () => await Package.FromUrl(nameof(InvalidUrlWontWork)));

		[Test]
		public void LoadingNonGithubPackageWontWork() =>
			Assert.ThrowsAsync<Package.OnlyGithubDotComUrlsAreAllowedForNow>(async () =>
				await Package.FromUrl("https://google.com"));
		
		[Test]
		public void NoneAndBooleanAreAlwaysKnown()
		{
			var emptyPackage = new Package(nameof(NoneAndBooleanAreAlwaysKnown));
			Assert.That(emptyPackage.FindType(Base.None), Is.Not.Null);
			Assert.That(emptyPackage.FindType(Base.Boolean), Is.Not.Null);
			Assert.That(emptyPackage.FindType(Base.Number), Is.Null);
		}

		[Test]
		public async Task LoadStrictBaseTypes()
		{
			var strictPackage = await Package.FromUrl(Package.StrictUrl);
			var basePackage = strictPackage.GetSubPackage(nameof(Base));
			Assert.That(basePackage.FindDirectType(Base.Any), Is.Not.Null);
			Assert.That(basePackage.FindDirectType(Base.Number), Is.Not.Null);
			Assert.That(basePackage.FindDirectType(Base.App), Is.Not.Null);
		}
	}
}