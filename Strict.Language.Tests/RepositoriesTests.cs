//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Threading.Tasks;
//using NUnit.Framework;

//namespace Strict.Language.Tests
//{
//	public class RepositoriesTests
//	{
//		[Test]
//		public void InvalidPathWontWork() =>
//			Assert.ThrowsAsync<DirectoryNotFoundException>(async () =>
//				await repos.LoadFromPath(nameof(InvalidPathWontWork)));

//		private readonly Repositories repos = new Repositories(null);

//		[Test]
//		public void LoadingNonGithubPackageWontWork() =>
//			Assert.ThrowsAsync<Repositories.OnlyGithubDotComUrlsAreAllowedForNow>(async () =>
//				await repos.LoadFromUrl(new Uri("https://google.com")));

//		[Test]
//		public async Task LoadStrictBaseTypes()
//		{
//			var strictPackage = await repos.LoadFromUrl(Repositories.StrictUrl);
//			var basePackage = strictPackage.FindSubPackage(nameof(Base))!;
//			Assert.That(basePackage.FindDirectType(Base.Any), Is.Not.Null);
//			Assert.That(basePackage.FindDirectType(Base.Number), Is.Not.Null);
//			Assert.That(basePackage.FindDirectType(Base.App), Is.Not.Null);
//		}

//		[Test]
//		public async Task LoadingSameRepositoryAgainUsesCache()
//		{
//			var tasks = new List<Task<Package>>();
//			for (int index=0; index <  100; index++)
//				tasks.Add(repos.LoadFromUrl(Repositories.StrictUrl));
//			await Task.WhenAll(tasks);
//			foreach (var task in tasks)
//				Assert.That(task.Result, Is.EqualTo(tasks[0].Result));
//		}
//	}
//}
