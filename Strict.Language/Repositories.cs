using System.Runtime.CompilerServices;
using LazyCache;

[assembly: InternalsVisibleTo("Strict.Compiler.Tests")]

namespace Strict.Language;

/// <summary>
/// Loads packages from url (like GitHub) and caches it to disc for the current and later runs.
/// Next time Repositories is created, we will check for outdated cache and delete the zip files
/// to allow redownloading fresh files. All locally cached packages and all types in them are
/// always available for any .strict file in the Editor. If a type is not found, we check on github
/// </summary>
/// <remarks>Everything in here is async, you can load many packages in parallel</remarks>
public sealed class Repositories
{
	/// <summary>
	/// Keeps a cache of loaded repositories for 20 minutes, default CachingService.DefaultCachePolicy
	/// </summary>
	public Repositories(ExpressionParser parser)
	{
		cacheService = new CachingService();
		this.parser = parser;
	}

	private readonly IAppCache cacheService;
	private readonly ExpressionParser parser;
	public Task<Package> LoadStrictPackage(string packageSubfolder = ""
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	) =>
#if DEBUG
		LoadFromUrl(new Uri(GitHubStrictUri.AbsoluteUri + (packageSubfolder == ""
			? ""
			: "/" + packageSubfolder)), callerFilePath, callerLineNumber, callerMemberName);
#else
		LoadFromUrl(new Uri(GitHubStrictUri.AbsoluteUri + (packageSubfolder == ""
			? ""
			: "/" + packageSubfolder)));
#endif

	public async Task<Package> LoadFromUrl(Uri packageUrl
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	)
	{
		var parts = packageUrl.AbsoluteUri.Split('/');
		if (parts.Length < 5 || parts[0] != "https:" || parts[1] != "" || parts[2] != "github.com")
			throw new OnlyGithubDotComUrlsAreAllowedForNow(packageUrl.AbsoluteUri);
		var organization = parts[3];
		var packageFullName = parts[4..].ToWordList(Context.ParentSeparator.ToString());
		var localDevelopmentPath = GetLocalDevelopmentPath(organization, packageFullName);
		if (Directory.Exists(localDevelopmentPath))
			return await LoadFromPath(packageFullName, localDevelopmentPath
#if DEBUG
				, callerFilePath, callerLineNumber, callerMemberName
#endif
			);
		var localCachePath = Path.Combine(CacheFolder, organization, packageFullName);
		if (PreviouslyCheckedDirectories.Add(localCachePath) && !Directory.Exists(localCachePath))
			await DownloadRepositoryStrictFiles(localCachePath, organization, packageFullName);
		return await LoadFromPath(packageFullName, localCachePath
#if DEBUG
			, callerFilePath, callerLineNumber, callerMemberName
#endif
		);
	}

	public static string GetLocalDevelopmentPath(string organization, string packageFullName) =>
		DevelopmentBaseFolder + organization + Context.ParentSeparator + packageFullName;

	public sealed class OnlyGithubDotComUrlsAreAllowedForNow(string uri) : Exception(uri +
		" is invalid. Valid url: " + GitHubStrictUri + ", it must always start with " +
		"https://github.com and only include the organization and repo name, nothing else!");

	//ncrunch: no coverage start, only called once per session and only if not on development machine
	private static readonly HashSet<string> PreviouslyCheckedDirectories = new();

	internal static async Task DownloadRepositoryStrictFiles(string localCachePath, string org,
		string repoNameAndOptionalSubFolders)
	{
		if (!Directory.Exists(localCachePath))
			Directory.CreateDirectory(localCachePath);
		using var downloader = new GitHubStrictDownloader(org, repoNameAndOptionalSubFolders);
		await downloader.DownloadFiles(localCachePath);
	} //ncrunch: no coverage end

	public Task<Package> LoadFromPath(string fullName, string packagePath
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	) =>
		cacheService.GetOrAddAsync(fullName, _ => CreatePackageFromFiles(packagePath,
			Directory.GetFiles(packagePath, "*" + Type.Extension)
#if DEBUG
			, null, callerFilePath, callerLineNumber, callerMemberName));
#else
		));
#endif

	/// <summary>
	/// Initially we need to create just empty types, and then after they all have been created,
	/// we will fill and load them, otherwise we could not use types within the package context.
	/// </summary>
	private async Task<Package> CreatePackageFromFiles(string packagePath,
		IReadOnlyCollection<string> files, Package? parent = null
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	)
	{
		// The main folder can be empty, other folders must contain at least one file to create a package
		if (parent != null && files.Count == 0)
			return parent; //ncrunch: no coverage
#if DEBUG
		var package = parent != null
			? new Package(parent, packagePath, this, callerFilePath, callerLineNumber, callerMemberName)
			: new Package(packagePath, this, callerFilePath, callerLineNumber, callerMemberName);
#else
		var package = parent != null
			? new Package(parent, packagePath, this)
			: new Package(packagePath, this);
#endif
		loadedPackages.Add(package);
		var types = GetTypes(files, package);
		foreach (var type in types)
			type.ParseMembersAndMethods(parser);
		return package;
	}

	private List<Package> loadedPackages = [];

	private ICollection<Type> GetTypes(IReadOnlyCollection<string> files, Package package)
	{
		var types = new List<Type>(files.Count);
		var filesWithMembers = new Dictionary<string, TypeLines>(StringComparer.Ordinal);
		foreach (var filePath in files)
		{
			var lines = new TypeLines(Path.GetFileNameWithoutExtension(filePath),
				File.ReadAllLines(filePath));
			if (lines.Name != Type.Mutable && lines.DependentTypes.Count > 0)
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
			var reversedDependencies = EmptyDegreeQueueAndGenerateSortedOutput(files, inDegreeGraphMap);
			if (inDegreeGraphMap.Any(keyValue => keyValue.Value > 0))
				AddUnresolvedRemainingTypes(files, inDegreeGraphMap, reversedDependencies);
			return reversedDependencies;
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

	private static Stack<TypeLines> EmptyDegreeQueueAndGenerateSortedOutput(
		IReadOnlyDictionary<string, TypeLines> files, Dictionary<string, int> inDegree)
	{
		var reversedDependencies = new Stack<TypeLines>();
		var zeroDegreeQueue = CreateZeroDegreeQueue(inDegree);
		while (zeroDegreeQueue.Count > 0)
			if (files.TryGetValue(zeroDegreeQueue.Dequeue(), out var lines))
			{
				reversedDependencies.Push(lines);
				foreach (var vertex in lines.DependentTypes)
					if (--inDegree[vertex] is 0)
						zeroDegreeQueue.Enqueue(vertex);
			}
		return reversedDependencies;
	}

	private static void AddUnresolvedRemainingTypes(IReadOnlyDictionary<string, TypeLines> files,
		Dictionary<string, int> inDegree, Stack<TypeLines> reversedDependencies)
	{
		foreach (var unresolvedType in inDegree.Where(x => x.Value > 0))
			if (files.TryGetValue(unresolvedType.Key, out var lines))
				if (reversedDependencies.All(
					alreadyAddedType => alreadyAddedType.Name != unresolvedType.Key))
					reversedDependencies.Push(lines);
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
/*user should specify which packages he wants, we shouldn't just load all in a repo!
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
	/// .vs, or _NCrunch) or numbers or dot separators (like Strict.Compiler) are allowed.
	/// </summary>
	private static bool IsValidCodeDirectory(string directory) =>
		Path.GetFileName(directory).IsWord();
*/
	public const string DevelopmentBaseFolder = @"C:\code\GitHub\";
	internal static string CacheFolder =>
		Path.Combine( //ncrunch: no coverage, only downloaded and cached on non-development machines
			Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), StrictPackages);
	private const string StrictPackages = nameof(StrictPackages);
	public const string StrictOrg = "strict-lang";
	public static readonly Uri GitHubStrictUri =
		new("https://github.com/" + StrictOrg + "/" + nameof(Strict));

	/// <summary>
	/// Called by Package.Dispose
	/// </summary>
	internal void Remove(Package result)
	{
		cacheService.Remove(result.FullName);
		loadedPackages.Remove(result);
	}

	public bool ContainsPackageNameInCache(string fullName) =>
		cacheService.TryGetValue<LazyCache.AsyncLazy<Package>>(fullName, out _);

	public async Task<string> ToDebugString() =>
		nameof(Repositories) +
		"\nStrict: " + (cacheService.TryGetValue<AsyncLazy<Package>>(nameof(Strict),
			out var lazyPackage)
			? (await lazyPackage.Value).ToDebugString()
			: "") +
		"\nLoadedPackages: " + loadedPackages.ToWordList("\n  ") +
		"\nPreviouslyCheckedDirectories: " + PreviouslyCheckedDirectories.ToWordList();
}