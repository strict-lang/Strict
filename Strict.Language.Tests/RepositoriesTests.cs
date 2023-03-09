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
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
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
		var basePackage = await repos.LoadStrictPackage();
		Assert.That(basePackage.FindDirectType(Base.Any), Is.Not.Null);
		Assert.That(basePackage.FindDirectType(Base.Number), Is.Not.Null);
		Assert.That(basePackage.FindDirectType(Base.App), Is.Not.Null);
	}

	[Test]
	public async Task LoadingSameRepositoryAgainUsesCache()
	{
		var tasks = new List<Task<Package>>();
		for (var index = 0; index < 10; index++)
			tasks.Add(repos.LoadStrictPackage());
		await Task.WhenAll(tasks);
		foreach (var task in tasks)
			Assert.That(task.Result, Is.EqualTo(tasks[0].Result));
	}

	[Test]
	public async Task MakeSureParsingFailedErrorMessagesAreClickable()
	{
		var parser = new MethodExpressionParser();
		var strictPackage = await new Repositories(parser).LoadStrictPackage();
		Assert.That(
			() => new Type(strictPackage,
				new TypeLines("Invalid", "has 1")).ParseMembersAndMethods(null!),
			Throws.InstanceOf<ParsingFailed>().With.Message.Contains(@"Base\Invalid.strict:line 1"));
	}

	[Test]
	public async Task LoadStrictExamplesPackageAndUseBasePackageTypes()
	{
		var parser = new MethodExpressionParser();
		var repositories = new Repositories(parser);
		await repositories.LoadStrictPackage();
		await repositories.LoadStrictPackage("Math");
		var examplesPackage = await repositories.LoadStrictPackage("Examples");
		var program = new Type(examplesPackage, new TypeLines("ValidProgram", "has number", "Run Number", "\tnumber")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ReturnType.ToString(), Contains.Substring(Base.Number));
		Assert.That(program.Members[0].Type.ToString(), Contains.Substring(Base.Number));
	}

	[Ignore("Fix parser issues with iterator first")]
	[Test]
	public async Task LoadStrictImageProcessingTypes()
	{
		var parser = new MethodExpressionParser();
		var repositories = new Repositories(parser);
		await repositories.LoadStrictPackage();
		var imageProcessingPackage =
			await repositories.LoadFromPath(
				StrictDevelopmentFolder + ".ImageProcessing");
		var adjustBrightness = imageProcessingPackage.GetType("AdjustBrightness");
		Assert.That(adjustBrightness, Is.Not.Null);
		Assert.That(adjustBrightness.Methods[0].GetBodyAndParseIfNeeded(), Is.Not.Null);
	}

	[Test]
	public async Task CheckGenericTypesAreLoadedCorrectlyAfterSorting()
	{
		var parser = new MethodExpressionParser();
		var repositories = new Repositories(parser);
		var program = new Type(await repositories.LoadStrictPackage(), new TypeLines("ValidProgram", "has texts", "Run Texts", "\t\"Result \" + 5")).
			ParseMembersAndMethods(parser);
		program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(program.Members[0].Type.IsIterator, Is.True);
		Assert.That(program.Members[0].Type.Members.Count, Is.GreaterThan(1));
		Assert.That(program.Members[0].Type.Methods.Count, Is.GreaterThan(5));
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
				new Repositories(null!).SortFilesByMemberUsage(CreateComplexImplementsDependencies()).
					Select(file => file.Name)), Is.EqualTo("File5, File6, File4, File3, File2, File1"));

	private static Dictionary<string, TypeLines> CreateComplexImplementsDependencies()
	{
		var file1 = new TypeLines("File1", "has File2");
		var file2 = new TypeLines("File2", "has File3", "has Number");
		var file3 = new TypeLines("File3", "has File4");
		var file4 = new TypeLines("File4", "has File5", "has File6");
		var file5 = new TypeLines("File5", "has Number");
		var file6 = new TypeLines("File6", "has File5");
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
	[Category("Manual")]
	[Test]
	public void NoFilesAllowedInStrictFolderNeedsToBeInASubFolder()
	{
		var strictFilePath = Path.Combine(StrictDevelopmentFolder,
			"UnitTestForCoverage.strict");
		File.Create(strictFilePath).Close();
		Assert.That(() => repos.LoadFromPath(StrictDevelopmentFolder),
			Throws.InstanceOf<Repositories.NoFilesAllowedInStrictFolderNeedsToBeInASubFolder>());
		File.Delete(strictFilePath);
	}

	public static readonly string StrictDevelopmentFolder = Repositories.StrictDevelopmentFolderPrefix[..^1];

	[Test]
	[Category("Slow")]
	[Benchmark]
	public void SortImplementsOneThousandTimes()
	{
		var files = CreateComplexImplementsDependencies();
		var repository = new Repositories(null!);
		for (var count = 0; count < 1000; count++)
			repository.SortFilesByMemberUsage(files);
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
				repository.SortFilesByMemberUsage(files);
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

	private static string BaseFolder => Repositories.StrictDevelopmentFolderPrefix + nameof(Base);

	[Test]
	[Category("Slow")]
	[Benchmark]
	public async Task LoadingZippedStrictBaseHundredTimes()
	{
		var zipFilePath = Path.Combine(StrictDevelopmentFolder, "Base.zip");
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
		//await repos.LoadStrictPackage();
		//MemoryProfiler.GetSnapshot(nameof(LoadStrictBaseTypesTenTimes) + " once");
		for (var iteration = 0; iteration < 100; iteration++)
			await repos.LoadStrictPackage();
		//MemoryProfiler.GetSnapshot(nameof(LoadStrictBaseTypesTenTimes) + "10");
	} //ncrunch: no coverage end
}