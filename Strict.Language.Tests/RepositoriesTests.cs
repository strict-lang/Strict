using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public sealed class RepositoriesTests
{
	[Test]
	public void InvalidPathWontWork() =>
		Assert.ThrowsAsync<DirectoryNotFoundException>(async () =>
			await repos.LoadFromPath(nameof(InvalidPathWontWork)));

	private readonly Repositories repos = new(new ExpressionParserTests());

	[Test]
	public void LoadingNonGithubPackageWontWork() =>
		Assert.ThrowsAsync<Repositories.OnlyGithubDotComUrlsAreAllowedForNow>(async () =>
			await repos.LoadFromUrl(new Uri("https://google.com")));

	[Test]
	public async Task LoadStrictBaseTypes()
	{
		var strictPackage = await repos.LoadFromUrl(Repositories.StrictUrl);
		var basePackage = strictPackage.FindSubPackage(nameof(Base))!;
		Assert.That(basePackage.FindDirectType(Base.Any), Is.Not.Null);
		Assert.That(basePackage.FindDirectType(Base.Number), Is.Not.Null);
		Assert.That(basePackage.FindDirectType(Base.App), Is.Not.Null);
	}

	[Test]
	public async Task LoadingSameRepositoryAgainUsesCache()
	{
		var tasks = new List<Task<Package>>();
		for (var index = 0; index < 20; index++)
			tasks.Add(repos.LoadFromUrl(Repositories.StrictUrl));
		await Task.WhenAll(tasks);
		foreach (var task in tasks)
			Assert.That(task.Result, Is.EqualTo(tasks[0].Result));
	}

	[Test]
	public async Task MakeSureParsingFailedErrorMessagesAreClickable()
	{
		var parser = new MethodExpressionParser();
		var strictPackage = await new Repositories(parser).LoadFromUrl(Repositories.StrictUrl);
		Assert.That(
			() => new Type(strictPackage.FindSubPackage("Examples")!, "Invalid", parser).Parse("has 1"),
			Throws.InstanceOf<ParsingFailed>().With.Message.Contains(@"at Strict.Examples.Invalid in " +
				Repositories.DevelopmentFolder + @"\Examples\Invalid.strict:line 1"));
	}
}