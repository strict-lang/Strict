using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace Strict.Language.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
public class RepositoriesTests
{
	[SetUp]
	public void CreateRepositories()
	{
		parser = new ExpressionParserTests();
		parser.CreateType();
		repos = new Repositories(parser);
	}

	private Repositories repos = null!;
	private ExpressionParserTests parser = null!;

	[TearDown]
	public void DisposeParserType() => parser.TearDown();

	[Test]
	public void InvalidPathWontWork() =>
		Assert.ThrowsAsync<DirectoryNotFoundException>(() =>
			repos.LoadFromPath(nameof(InvalidPathWontWork), nameof(InvalidPathWontWork)));

	[Test]
	public void LoadingNonGithubPackageWontWork() =>
		Assert.ThrowsAsync<Repositories.OnlyGithubDotComUrlsAreAllowedForNow>(() =>
			repos.LoadFromUrl(new Uri("https://google.com")));

	[Test]
	public async Task LoadStrictBaseTypes()
	{
		Assert.That(repos.ContainsPackageNameInCache(nameof(Strict)), Is.False,
			await repos.ToDebugString());
		using var basePackage = await repos.LoadStrictPackage();
		Assert.That(basePackage.FindDirectType(Type.Any), Is.Not.Null);
		Assert.That(basePackage.FindDirectType(Type.Number), Is.Not.Null);
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
		tasks[0].Result.Dispose();
	}

	[Test]
	public async Task MakeSureParsingFailedErrorMessagesAreClickable()
	{
		using var strictPackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		Assert.That(() =>
			{
				using var _ = new Type(strictPackage, new TypeLines("Invalid", "has 1")).
					ParseMembersAndMethods(null!);
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.Message.
				Contains(Path.Combine("Strict", "Invalid.strict") + ":line 1"));
	}

	[Test]
	public async Task LoadStrictExamplesPackageAndUseBasePackageTypes()
	{
		using var basePackage = await repos.LoadStrictPackage();
		using var mathPackage = await repos.LoadStrictPackage("Strict/Math");
		using var examplesPackage = await repos.LoadStrictPackage("Strict/Examples");
		using var program =
			new Type(examplesPackage,
					new TypeLines("ValidProgram", "has number", "Run Number", "\tnumber")).
				ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ReturnType.ToString(), Contains.Substring(Type.Number));
		Assert.That(program.Members[0].Type.ToString(), Contains.Substring(Type.Number));
	}

	[Test]
	public async Task LoadStrictImageProcessingTypes()
	{
		using var package =
			await repos.LoadStrictPackage(nameof(Strict) + Context.ParentSeparator + "ImageProcessing");
		var adjustBrightness = package.GetType("AdjustBrightness");
		Assert.That(adjustBrightness, Is.Not.Null);
		Assert.That(adjustBrightness.Methods[0].GetBodyAndParseIfNeeded(), Is.Not.Null);
	}

	[Test]
	public async Task CheckGenericTypesAreLoadedCorrectlyAfterSorting()
	{
		using var package = await repos.LoadStrictPackage();
		using var program = new Type(package,
				new TypeLines("ValidProgram", "has texts", "Run Texts", "\t\"Result \" + 5")).
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
	///			File5 needs Number
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
	/// impact on how we load the files, parallel or not, async is only 10-20% faster and not
	/// important. File.ReadAllLinesAsync is by far the slowest way (2-3x slower) to load files.
	/// </summary>
	[Test]
	[Category("Slow")]
	[Benchmark]
	public void LoadingAllStrictFilesWithoutAsyncHundredTimes()
	{
		for (var iteration = 0; iteration < 100; iteration++)
			foreach (var file in Directory.GetFiles(
				Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
				"*" + Type.Extension))
				File.ReadAllLines(file);
	}

	[Test]
	[Category("Slow")]
	[Benchmark]
	public async Task LoadStrictBaseTypesHundredTimes()
	{
#if MEMORY_PROFILER
		await repos.LoadStrictPackage();
		// Requires Jetbrains.Profiler.Windows.Api.MemoryProfiler
		MemoryProfiler.GetSnapshot(nameof(LoadStrictBaseTypesHundredTimes) + " once");
#endif
		Package? package = null;
		for (var iteration = 0; iteration < 100; iteration++)
			package = await repos.LoadStrictPackage();
#if MEMORY_PROFILER
		MemoryProfiler.GetSnapshot(nameof(LoadStrictBaseTypesHundredTimes) + " 100 more times");
#endif
		package!.Dispose();
	}

	[Test]
	[Category("Manual")]
	public void BenchmarkIsOperator() => BenchmarkRunner.Run<BinaryOperatorTests>();

	[Test]
	public async Task LoadStrictBaseTextTypeContainsNewRuntimeMethods()
	{
		using var package = await repos.LoadStrictPackage();
		var textType = package.GetType(Type.Text);
		Assert.That(textType.Methods.Any(method => method.Name == "Split"), Is.True);
		Assert.That(textType.Methods.Any(method => method.Name == "Trim"), Is.True);
		Assert.That(textType.Methods.Any(method => method.Name == "IndexOf"), Is.True);
		Assert.That(textType.Methods.Any(method => method.Name == "LastIndexOf"), Is.True);
		Assert.That(textType.Methods.Any(method => method.Name == "Substring"), Is.True);
		Assert.That(textType.Methods.Any(method => method.Name == "Replace"), Is.True);
		Assert.That(textType.Methods.Any(method => method.Name == "Upper"), Is.True);
		Assert.That(textType.Methods.Any(method => method.Name == "Lower"), Is.True);
	}

	[Test]
	public async Task LoadStrictBaseTypesContainPathDirectoryAndFileRuntimeMethods()
	{
		using var package = await repos.LoadStrictPackage();
		var pathType = package.GetType("Path");
		Assert.That(pathType.Methods.Any(method => method.Name == "+"), Is.True);
		Assert.That(pathType.Methods.Any(method => method.Name == "from"), Is.False);
		Assert.That(pathType.Methods.Any(method => method.Name == "to"), Is.False);
		Assert.That(pathType.Methods.Any(method => method.Name == "FileName"), Is.True);
		Assert.That(pathType.Methods.Any(method => method.Name == "RemoveExtension"), Is.True);
		Assert.That(pathType.Methods.Any(method => method.Name == "ChangeExtension"), Is.True);
		Assert.That(pathType.Methods.Any(method => method.Name == "PathOnly"), Is.True);
		Assert.That(pathType.Methods.Any(method => method.Name == "FileNameWithoutExtension"), Is.False);
		Assert.That(pathType.Methods.Any(method => method.Name == "DirectoryName"), Is.False);
		Assert.That(pathType.Methods.Single(method => method.Name == "FileName").ReturnType.Name,
			Is.EqualTo("Path"));
		Assert.That(pathType.Methods.Single(method => method.Name == "RemoveExtension").ReturnType.Name,
			Is.EqualTo("Path"));
		Assert.That(pathType.Methods.Single(method => method.Name == "PathOnly").ReturnType.Name,
			Is.EqualTo("Path"));
		var directoryType = package.GetType("Directory");
		Assert.That(directoryType.Methods.Any(method => method.Name == "Exists"), Is.True);
		Assert.That(directoryType.Methods.Any(method => method.Name == "Files"), Is.True);
		Assert.That(directoryType.Methods.Any(method => method.Name == "Create"), Is.True);
		var fileType = package.GetType(Type.File);
		Assert.That(fileType.Methods.Any(method => method.Name == "Exists"), Is.True);
	}

	[Test]
	public async Task LoadStrictCharacterTypeContainsCaseMethods()
	{
		using var package = await repos.LoadStrictPackage();
		var characterType = package.GetType(Type.Character);
		Assert.That(characterType.Methods.Any(method => method.Name == "Upper"), Is.True);
		Assert.That(characterType.Methods.Any(method => method.Name == "Lower"), Is.True);
	}
}