using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, targetCount: 10)]
public class RepositoriesTests
{
	[Test]
	public void InvalidPathWontWork() =>
		Assert.ThrowsAsync<DirectoryNotFoundException>(() =>
			repos.LoadFromPath(nameof(InvalidPathWontWork)));

	private readonly Repositories repos = new(new ExpressionParserTests());

	[Test]
	public void LoadingNonGithubPackageWontWork() =>
		Assert.ThrowsAsync<Repositories.OnlyGithubDotComUrlsAreAllowedForNow>(() =>
			repos.LoadFromUrl(new Uri("https://google.com")));

	[Test]
	public async Task LoadStrictBaseTypes()
	{
		var strictPackage = await repos.LoadFromUrl(Repositories.StrictUrl);
		var basePackage = strictPackage.FindSubPackage(nameof(Base))!;
		Assert.That(basePackage.FindDirectType(Base.Any), Is.Not.Null);
		Assert.That(basePackage.FindDirectType(Base.Number), Is.Not.Null);
		Assert.That(basePackage.FindDirectType(Base.App), Is.Not.Null);
	}

	[Category("Manual")]
	[Test]
	public void NoFilesAllowedInStrictFolderNeedsToBeInASubFolder()
	{
		var strictFilePath = Path.Combine(Repositories.DevelopmentFolder, "UnitTestForCoverage.strict");
		File.Create(strictFilePath).Close();
		Assert.That(() =>
			repos.LoadFromPath(Repositories.DevelopmentFolder), Throws.InstanceOf<Repositories.NoFilesAllowedInStrictFolderNeedsToBeInASubFolder>());
		File.Delete(strictFilePath);
	}

	[Test]
	public async Task LoadingSameRepositoryAgainUsesCache()
	{
		var tasks = new List<Task<Package>>();
		for (var index = 0; index < 10; index++)
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
			() => new Type(strictPackage.FindSubPackage("Examples")!,
				new TypeLines("Invalid", "has 1")).ParseMembersAndMethods(null!),
			Throws.InstanceOf<ParsingFailed>().With.Message.Contains(@"at Strict.Examples.Invalid in " +
				Repositories.DevelopmentFolder + @"\Examples\Invalid.strict:line 1"));
	}

	/// <summary>
	/// Each indentation is one depth level lower
	/// File1 needs File2
	///  File2 needs File3 and Number
	///   File3 needs File4
	///    File4 needs File5 and File6
	///			 File5 needs Number
	/// File6 needs File5
	/// </summary>
	[Test]
	public void SortImplements() =>
		Assert.That(
			string.Join(", ",
				new Repositories(null!).SortFilesWithImplements(CreateComplexImplementsDependencies()).
					Select(file => file.Name)), Is.EqualTo("File5, File4, File3, File2, File6, File1"));

	private static Dictionary<string, TypeLines> CreateComplexImplementsDependencies()
	{
		var file1 = new TypeLines("File1", "implement File2");
		var file2 = new TypeLines("File2", "implement File3", "implement Number");
		var file3 = new TypeLines("File3", "implement File4");
		var file4 = new TypeLines("File4", "implement File5", "implement File 6");
		var file5 = new TypeLines("File5", "implement Number");
		var file6 = new TypeLines("File6", "implement File5");
		var filesWithImplements = new Dictionary<string, TypeLines>(StringComparer.Ordinal)
		{
			{ file1.Name, file1 },
			{ file2.Name, file2 },
			{ file3.Name, file3 },
			{ file4.Name, file4 },
			{ file5.Name, file5 },
			{ file6.Name, file6 }
		};
		return filesWithImplements;
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	[Benchmark]
	public void SortImplementsOneThousandTimes()
	{
		var files = CreateComplexImplementsDependencies();
		var repository = new Repositories(null!);
		for (var count = 0; count < 1000; count++)
			repository.SortFilesWithImplements(files);
	}

	[Test]
	[Category("Slow")]
	[Benchmark]
	public void SortImplementsOneThousandTimesInParallel()
	{
		var files = CreateComplexImplementsDependencies();
		var repository = new Repositories(null!);
		Parallel.For(0, 12, (_, _) =>
		{
			for (var count = 0; count < 1000; count++)
				repository.SortFilesWithImplements(files);
		});
	}

	/// <summary>
	/// Zip file loading makes a difference (4-5 times faster), but otherwise there is close to zero
	/// impact how we load the files, parallel or not, async is only 10-20% faster and not important.
	/// File.ReadAllLinesAsync is by far the slowest way (2-3x slower) to load files.
	/// </summary>
	[Test]
	[Category("Slow")]
	[Benchmark]
	public void LoadingAllStrictFilesWithoutAsyncHundredTimes()
	{
		for (var iteration = 0; iteration < 100; iteration++)
			foreach (var file in Directory.GetFiles(BaseFolder, "*.strict"))
				File.ReadAllLines(file);
	}

	private static string BaseFolder => Path.Combine(Repositories.DevelopmentFolder, "Base");

	[Test]
	[Category("Slow")]
	[Benchmark]
	public async Task LoadingZippedStrictBaseHundredTimes()
	{
		var zipFilePath = Path.Combine(Repositories.DevelopmentFolder, "Base.zip");
		if (!File.Exists(zipFilePath))
			ZipFile.CreateFromDirectory(BaseFolder, zipFilePath);
		for (var iteration = 0; iteration < 100; iteration++)
		{
			var tasks = new List<Task>();
			foreach (var entry in ZipFile.OpenRead(zipFilePath).Entries)
				tasks.Add(new StreamReader(entry.Open()).ReadToEndAsync());
			await Task.WhenAll(tasks);
		}
	}

	[Test]
	[Category("Slow")]
	[Benchmark]
	public async Task LoadStrictBaseTypesHundredTimes()
	{
		//await repos.LoadFromUrl(Repositories.StrictUrl);
		//MemoryProfiler.GetSnapshot(nameof(LoadStrictBaseTypesTenTimes) + " once");
		for (var iteration = 0; iteration < 100; iteration++)
			await repos.LoadFromUrl(Repositories.StrictUrl);
		//MemoryProfiler.GetSnapshot(nameof(LoadStrictBaseTypesTenTimes) + "10");
	}
}