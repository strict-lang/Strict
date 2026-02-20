using System.IO.Compression;
using System.Runtime.CompilerServices;
using LazyCache;

[assembly: InternalsVisibleTo("Strict.Compiler.Tests")]

namespace Strict.Language;

/// <summary>
/// Loads packages from url (like GitHub) and caches it to disc for the current and subsequent
/// runs. Next time Repositories is created, we will check for outdated cache and delete the zip
/// files to allow redownloading fresh files. All locally cached packages and all types in them
/// are always available for any .strict file in the Editor. If a type is not found,
/// packages.strict.dev is asked if we can get a url (used here to load).
/// </summary>
/// <remarks>Everything in here is async, you can easily load many packages in parallel</remarks>
public sealed class Repositories
{
	/// <summary>
	/// Gets rid of any cached zip files (keeps the actual files for use) older than 1h, which will
	/// allow redownloading from GitHub to get any changes, while still staying fast in local runs
	/// when there are usually no changes happening.
	/// </summary>
	public Repositories(ExpressionParser parser)
	{
		cacheService = new CachingService();
		this.parser = parser;
		if (Directory.Exists(CacheFolder))
			//ncrunch: no coverage start, rarely happens
			foreach (var file in Directory.GetFiles(CacheFolder, "*.zip"))
				if (File.GetLastWriteTimeUtc(file) < DateTime.UtcNow.AddHours(-1))
					File.Delete(file);
	} //ncrunch: no coverage end

	private readonly IAppCache cacheService;
	private readonly ExpressionParser parser;

	public async Task<Package> LoadFromUrl(Uri packageUrl
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	)
	{
		var isStrictPackage = packageUrl.AbsoluteUri.StartsWith(StrictPrefixUri.AbsoluteUri, StringComparison.Ordinal);
		if (!isStrictPackage && (packageUrl.Host != "github.com" || string.IsNullOrEmpty(packageUrl.AbsolutePath)))
			throw new OnlyGithubDotComUrlsAreAllowedForNow();
		var packageName = packageUrl.AbsolutePath.Split('/').Last();
		if (isStrictPackage)
		{
			var developmentFolder =
				StrictDevelopmentFolderPrefix.Replace(nameof(Strict) + ".", packageName);
			if (Directory.Exists(developmentFolder))
				return await LoadFromPath(developmentFolder
#if DEBUG
					// ReSharper disable ExplicitCallerInfoArgument
					, callerFilePath, callerLineNumber, callerMemberName
#endif
				);
		} //ncrunch: no coverage
		return await FindOrAddPath(packageUrl, packageName); //ncrunch: no coverage
	}

	private async Task<Package> FindOrAddPath(Uri packageUrl, string packageName)
	{ //ncrunch: no coverage start
		var localPath = Path.Combine(CacheFolder, packageName);
		if (!PreviouslyCheckedDirectories.Add(localPath))
			return await LoadFromPath(localPath);
		if (!Directory.Exists(localPath))
			localPath = await DownloadAndExtractRepository(packageUrl, packageName);
		return await LoadFromPath(localPath);
	} //ncrunch: no coverage end

	public Task<Package> LoadStrictPackage(string packagePostfixName = nameof(Base)
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	) =>
		LoadFromUrl(new Uri(StrictPrefixUri.AbsoluteUri + packagePostfixName), callerFilePath,
			callerLineNumber, callerMemberName);

	public sealed class OnlyGithubDotComUrlsAreAllowedForNow : Exception;
	//ncrunch: no coverage start, only called once per session and only if not on development machine
	private static readonly HashSet<string> PreviouslyCheckedDirectories = new();

	internal static async Task<string> DownloadAndExtractRepository(Uri packageUrl,
		string packageName)
	{
		if (!Directory.Exists(CacheFolder))
			Directory.CreateDirectory(CacheFolder);
		var targetPath = Path.Combine(CacheFolder, packageName);
		if (Directory.Exists(targetPath) &&
			File.Exists(Path.Combine(CacheFolder, packageName + ".zip")))
			return targetPath;
		await DownloadAndExtract(packageUrl, packageName, targetPath);
		return targetPath;
	}

	private static async Task DownloadAndExtract(Uri packageUrl, string packageName,
		string targetPath)
	{
		var localZip = Path.Combine(CacheFolder, packageName + ".zip");
		using HttpClient client = new();
		await DownloadFile(client, new Uri(packageUrl + "/archive/master.zip"), localZip);
		await Task.Run(() =>
			UnzipInCacheFolderAndMoveToTargetPath(packageName, targetPath, localZip));
	}

	private static async Task DownloadFile(HttpClient client, Uri uri, string fileName)
	{
		await using var stream = await client.GetStreamAsync(uri);
		await using var file = new FileStream(fileName, FileMode.CreateNew);
		await stream.CopyToAsync(file);
	}

	private static void UnzipInCacheFolderAndMoveToTargetPath(string packageName, string targetPath,
		string localZip)
	{
		ZipFile.ExtractToDirectory(localZip, CacheFolder, true);
		var masterDirectory = Path.Combine(CacheFolder, packageName + "-master");
		if (!Directory.Exists(masterDirectory))
			throw new NoMasterFolderFoundFromPackage(packageName, localZip);
		if (Directory.Exists(targetPath))
			new DirectoryInfo(targetPath).Delete(true);
		TryMoveOrCopyWhenDeletionDidNotFullyWork(targetPath, masterDirectory);
	}

	public sealed class NoMasterFolderFoundFromPackage(string packageName, string localZip)
		: Exception(packageName + ", localZip: " + localZip);

	private static void TryMoveOrCopyWhenDeletionDidNotFullyWork(string targetPath,
		string masterDirectory)
	{
		try
		{
			Directory.Move(masterDirectory, targetPath);
		}
		catch
		{
			foreach (var file in Directory.GetFiles(masterDirectory))
				File.Copy(file, Path.Combine(targetPath, Path.GetFileName(file)), true);
		}
	} //ncrunch: no coverage end

	public Task<Package> LoadFromPath(string packagePath
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	) =>
		cacheService.GetOrAddAsync(packagePath,
			_ => CreatePackageFromFiles(packagePath,
				// ReSharper disable ExplicitCallerInfoArgument
				Directory.GetFiles(packagePath, "*" + Type.Extension), null, callerFilePath,
				callerLineNumber, callerMemberName));

	/// <summary>
	/// Initially we need to create just empty types, and then after they all have been created,
	/// we will fill and load them, otherwise we could not use types within the package context.
	/// </summary>
	private async Task<Package> CreatePackageFromFiles(string packagePath,
		IReadOnlyCollection<string> files,
#if DEBUG
		Package? parent = null, [CallerFilePath] string callerFilePath = "",
		[CallerLineNumber] int callerLineNumber = 0, [CallerMemberName] string callerMemberName = "")
#else
		Package? parent = null)
#endif
	{
		// The main folder can be empty, other folders must contain at least one file to create a package
		if (parent != null && files.Count == 0)
			return parent; //ncrunch: no coverage
#if DEBUG
		var folderName = Path.GetFileName(packagePath);
		var package = parent != null
			// ReSharper disable ExplicitCallerInfoArgument
			? new Package(parent, packagePath, callerFilePath, callerLineNumber, callerMemberName)
			: new Package(folderName.Contains('.')
				? folderName.Split('.')[1]
				: packagePath, callerFilePath, callerLineNumber, callerMemberName);
#else
		var folderName = Path.GetFileName(packagePath);
		var package = parent != null
			? new Package(parent, packagePath)
			: new Package(folderName.Contains('.')
				? folderName.Split('.')[1]
				: packagePath);
#endif
		if (package.Name == nameof(Strict) && files.Count > 0)
			throw new NoFilesAllowedInStrictFolderNeedsToBeInASubFolder(files); //ncrunch: no coverage
		var types = GetTypes(files, package);
		foreach (var type in types)
			type.ParseMembersAndMethods(parser);
		await GetSubDirectoriesAndParse(packagePath, package
#if DEBUG
			, callerFilePath, callerLineNumber, callerMemberName
#endif
		);
		return package;
	}

	//ncrunch: no coverage start
	public sealed class NoFilesAllowedInStrictFolderNeedsToBeInASubFolder(IEnumerable<string> files)
		: Exception(files.ToWordList()); //ncrunch: no coverage end

	private ICollection<Type> GetTypes(IReadOnlyCollection<string> files, Package package)
	{
		var types = new List<Type>(files.Count);
		var filesWithMembers = new Dictionary<string, TypeLines>(StringComparer.Ordinal);
		foreach (var filePath in files)
		{
			var lines = new TypeLines(Path.GetFileNameWithoutExtension(filePath),
				File.ReadAllLines(filePath));
			if (lines.Name != Base.Mutable && lines.DependentTypes.Count > 0)
				filesWithMembers.Add(lines.Name, lines);
			else
				types.Add(new Type(package, lines));
		}
		return GetTypesFromSortedFiles(types, SortFilesByMemberUsage(filesWithMembers), package);
	}

	/// <summary>
	/// https://en.wikipedia.org/wiki/Breadth-first_search
	/// </summary>
	public IEnumerable<TypeLines> SortFilesByMemberUsage(Dictionary<string, TypeLines> files)
	{
		var inDegreeGraphMap = CreateInDegreeGraphMap(files);
		if (GotNestedImplements(files))
		{
			var sortedDependencies = EmptyDegreeQueueAndGenerateSortedOutput(files, inDegreeGraphMap);
			if (inDegreeGraphMap.Any(keyValue => keyValue.Value > 0))
				AddUnresolvedRemainingTypes(files, inDegreeGraphMap, sortedDependencies);
			return sortedDependencies;
		}
		return files.Values; //ncrunch: no coverage
	}

	private static bool GotNestedImplements(Dictionary<string, TypeLines> filesWithMembers)
	{
		foreach (var file in filesWithMembers)
			// ReSharper disable once ForCanBeConvertedToForeach, not done for performance reasons
			for (var index = 0; index < file.Value.DependentTypes.Count; index++)
				if (filesWithMembers.ContainsKey(file.Value.DependentTypes[index]))
					return true;
		return false; //ncrunch: no coverage
	}

	private static Dictionary<string, int> CreateInDegreeGraphMap(
		Dictionary<string, TypeLines> filesWithImplements)
	{
		var inDegree = new Dictionary<string, int>(StringComparer.Ordinal);
		foreach (var kvp in filesWithImplements)
		{
			inDegree.TryAdd(kvp.Key, 0);
			foreach (var edge in kvp.Value.DependentTypes)
				if (!inDegree.TryAdd(edge, 1))
					inDegree[edge]++;
		}
		return inDegree;
	}

	private static List<TypeLines> EmptyDegreeQueueAndGenerateSortedOutput(
		IReadOnlyDictionary<string, TypeLines> files, Dictionary<string, int> inDegree)
	{
		var sortedDependencies = new List<TypeLines>();
		var zeroDegreeQueue = CreateZeroDegreeQueue(inDegree);
		while (zeroDegreeQueue.Count > 0)
			if (files.TryGetValue(zeroDegreeQueue.Dequeue(), out var lines))
			{
				sortedDependencies.Add(lines);
				foreach (var vertex in lines.DependentTypes)
					if (--inDegree[vertex] is 0)
						zeroDegreeQueue.Enqueue(vertex);
			}
		return sortedDependencies;
	}

	private static void AddUnresolvedRemainingTypes(IReadOnlyDictionary<string, TypeLines> files,
		Dictionary<string, int> inDegree, List<TypeLines> sortedDependencies)
	{
		foreach (var unresolvedType in inDegree.Where(x => x.Value > 0))
			if (files.TryGetValue(unresolvedType.Key, out var lines))
				if (sortedDependencies.All(
					alreadyAddedType => alreadyAddedType.Name != unresolvedType.Key))
					sortedDependencies.Add(lines);
	}

	private static Queue<string> CreateZeroDegreeQueue(Dictionary<string, int> inDegree)
	{
		var zeroDegreeQueue = new Queue<string>();
		foreach (var vertex in inDegree)
			if (vertex.Value is 0)
				zeroDegreeQueue.Enqueue(vertex.Key);
		return zeroDegreeQueue;
	}

	private static ICollection<Type> GetTypesFromSortedFiles(ICollection<Type> types,
		IEnumerable<TypeLines> sortedFiles, Package package)
	{
		foreach (var typeLines in sortedFiles)
			types.Add(new Type(package, typeLines));
		return types;
	}

	private async Task GetSubDirectoriesAndParse(string packagePath, Package package
#if DEBUG
		, string callerFilePath, int callerLineNumber, string callerMemberName
#endif
	)
	{
		var subDirectories = Directory.GetDirectories(packagePath);
		if (subDirectories.Length > 0)
			await Task.WhenAll(ParseAllSubFolders(subDirectories, package
#if DEBUG
				, callerFilePath, callerLineNumber, callerMemberName
#endif
			));
	}

	private List<Task> ParseAllSubFolders(IEnumerable<string> subDirectories, Package package
#if DEBUG
		, string callerFilePath, int callerLineNumber, string callerMemberName
#endif
	)
	{
		var tasks = new List<Task>();
		foreach (var directory in subDirectories)
			if (IsValidCodeDirectory(directory))
				tasks.Add(CreatePackageFromFiles(directory, //ncrunch: no coverage
					Directory.GetFiles(directory, "*" + Type.Extension), package
#if DEBUG
					, callerFilePath, callerLineNumber, callerMemberName
#endif
				));
		return tasks;
	}

	/// <summary>
	/// In Strict only words are valid directory names = package names, no symbols (like .git, .hg,
	/// .vs or _NCrunch) or numbers or dot separators (like Strict.Compiler) are allowed.
	/// </summary>
	private static bool IsValidCodeDirectory(string directory) =>
		Path.GetFileName(directory).IsWord();

	public const string StrictDevelopmentFolderPrefix = @"C:\code\GitHub\strict-lang\Strict.";
	private static string CacheFolder =>
		Path.Combine( //ncrunch: no coverage, only downloaded and cached on non-development machines
			Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), StrictPackages);
	private const string StrictPackages = nameof(StrictPackages);
	public static readonly Uri StrictPrefixUri = new("https://github.com/strict-lang/Strict.");
}