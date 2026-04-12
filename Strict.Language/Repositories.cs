using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using LazyCache;

[assembly: InternalsVisibleTo("Strict.Transpiler.Tests")]

namespace Strict.Language;

/// <summary>
/// Loads packages from url (like GitHub) and caches it to disc for the current and later runs.
/// Next time Repositories is created, we will check for outdated cache and delete the zip files
/// to allow redownloading fresh files. All locally cached packages and all types in them are
/// always available for any .strict file in the Editor. If a type is not found, we check on github
/// </summary>
/// <remarks>Everything in here is async, you can load many packages in parallel</remarks>
public sealed class Repositories(ExpressionParser parser)
{
	public Task<Package> LoadStrictPackage(string packageNameAndSubfolder = nameof(Strict)
#if DEBUG
		, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = ""
#endif
	) =>
#if DEBUG
		LoadFromUrl(new Uri(GitHubStrictUri.AbsoluteUri + packageNameAndSubfolder), callerFilePath,
			callerLineNumber, callerMemberName);
#else
		LoadFromUrl(new Uri(GitHubStrictUri.AbsoluteUri + packageNameAndSubfolder));
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
		var remaining = parts[4..];
		var packageFullName = string.Join<string>(Context.ParentSeparator.ToString(), remaining);
		var localDevelopmentPath = GetLocalDevelopmentPath(organization, packageFullName);
		if (Directory.Exists(localDevelopmentPath))
			return await LoadFromPath(packageFullName, localDevelopmentPath
#if DEBUG
				, callerFilePath, callerLineNumber, callerMemberName
#endif
			);
		//ncrunch: no coverage start
		var localCachePath = Path.Combine(CacheFolder, organization, packageFullName);
		if (PreviouslyCheckedDirectories.Add(localCachePath) && !Directory.Exists(localCachePath))
			await DownloadRepositoryStrictFiles(localCachePath, organization, packageFullName);
		return await LoadFromPath(packageFullName, localCachePath
#if DEBUG
			, callerFilePath, callerLineNumber, callerMemberName
#endif
		);
	} //ncrunch: no coverage end

	private readonly IAppCache cacheService = new CachingService();
	private readonly ExpressionParser parser = parser;

	public static string GetLocalDevelopmentPath(string organization, string packageFullName)
	{
		var path = DevelopmentBaseFolder + organization + Context.ParentSeparator + packageFullName;
		if (Directory.Exists(path))
			return path;
		var repoRoot = FindRepositoryRoot();
		if (repoRoot == null)
			return path;
		// When running inside the Strict repo, the repo root IS the base Strict package
		// (contains .strict files). Sub-packages like "Strict/Math" map to repoRoot/Math.
		var separatorIndex = packageFullName.IndexOf(Context.ParentSeparator);
		var repoPath = separatorIndex < 0
			? repoRoot
			: repoRoot + Context.ParentSeparator + packageFullName[(separatorIndex + 1)..];
		return Directory.Exists(repoPath)
			? repoPath
			: repoRoot + Context.ParentSeparator + packageFullName;
	}

	private static string? FindRepositoryRoot()
	{
		if (cachedRepositoryRoot != null)
			return cachedRepositoryRoot;
		var current = AppContext.BaseDirectory;
		while (current != null)
		{
			if (File.Exists(Path.Combine(current, Type.Any + Type.Extension)))
				return cachedRepositoryRoot = current;
			current = Path.GetDirectoryName(current);
		}
		return null;
	}

	private static string? cachedRepositoryRoot;

	public sealed class OnlyGithubDotComUrlsAreAllowedForNow(string uri) : Exception(uri +
		" is invalid. Valid url: " + GitHubStrictUri + nameof(Strict) + ", it must always start " +
		"with https://github.com and only include the organization and repo name!");

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
	)
	{
		var files = Directory.GetFiles(packagePath, "*" + Type.Extension);
		return cacheService.GetOrAddAsync(fullName, async _ =>
		{
			var parent = await LoadParentPackage(fullName);
			await LoadDependencyPackages(fullName, files);
			return await CreatePackageFromFiles(packagePath, files
#if DEBUG
				, parent, callerFilePath, callerLineNumber, callerMemberName);
#else
				, parent);
#endif
		});
	}

	private async Task<Package?> LoadParentPackage(string fullName)
	{
		var parent = FindParentPackage(fullName);
		if (parent != null)
			return parent;
		var separatorIndex = fullName.LastIndexOf(Context.ParentSeparator);
		if (separatorIndex < 0)
			return null;
		var parentName = fullName[..separatorIndex];
		return await LoadStrictPackage(parentName);
	}

	private async Task LoadDependencyPackages(string fullName, IReadOnlyCollection<string> files)
	{
		if (fullName == nameof(Strict) || files.Count == 0)
			return;
		var rootPackageName = GetRootPackageName(fullName);
		foreach (var dependencyPackage in FindDependencyPackages(fullName, rootPackageName, files))
			await LoadStrictPackage(dependencyPackage);
	}

	private static string GetRootPackageName(string fullName)
	{
		var separatorIndex = fullName.IndexOf(Context.ParentSeparator);
		return separatorIndex == -1
			? fullName
			: fullName[..separatorIndex];
	}

	private static IEnumerable<string> FindDependencyPackages(string fullName, string rootPackageName,
		IReadOnlyCollection<string> files)
	{
		var dependencies = new HashSet<string>(StringComparer.Ordinal);
		foreach (var file in files)
		{
			foreach (Match match in TypeFullNamePattern.Matches(File.ReadAllText(file)))
			{
				var typeFullName = match.Value;
				var lastSeparatorIndex = typeFullName.LastIndexOf(Context.ParentSeparator);
				if (lastSeparatorIndex <= 0)
					continue;
				var packageName = NormalizePackageName(typeFullName[..lastSeparatorIndex], rootPackageName);
				if (packageName != fullName)
					dependencies.Add(packageName);
			}
		}
		return dependencies;
	}

	private static string NormalizePackageName(string packageName, string rootPackageName)
	{
		if (packageName == rootPackageName ||
			packageName.StartsWith(rootPackageName + Context.ParentSeparator, StringComparison.Ordinal))
			return packageName;
		return rootPackageName + Context.ParentSeparator + packageName;
	}

	private static readonly Regex TypeFullNamePattern = new(
		"(?<![A-Za-z0-9/])[A-Z][A-Za-z0-9]*(?:/[A-Z][A-Za-z0-9]*)+(?![A-Za-z0-9/])",
		RegexOptions.Compiled);

	private Package? FindParentPackage(string fullName)
	{
		var separatorIndex = fullName.LastIndexOf(Context.ParentSeparator);
		if (separatorIndex < 0)
			return null;
		var parentName = fullName[..separatorIndex];
		// ReSharper disable once InconsistentlySynchronizedField
		return loadedPackages.Find(package => package.FullName == parentName);
	}

	/// <summary>
	/// Initially we need to create just empty types, and then after they all have been created,
	/// we will fill and load them, otherwise we could not use types within the package context.
	/// Constraint parsing is deferred to a second pass so all type methods are available.
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
		lock (loadedPackages)
			loadedPackages.Add(package);
		var types = GetTypes(files, package);
		foreach (var type in types)
			type.ParseMembersAndMethodsForPackage(parser);
		InvalidateAllAvailableMethodsCaches();
		foreach (var type in types)
			type.ParseDeferredConstraints(parser);
		return package;
	}

	private void InvalidateAllAvailableMethodsCaches()
	{
		Package[] loadedPackagesSnapshot;
		lock (loadedPackages)
			loadedPackagesSnapshot = loadedPackages.ToArray();
		foreach (var loadedPackage in loadedPackagesSnapshot)
		foreach (var type in loadedPackage.Types.Values.ToArray())
			type.InvalidateAvailableMethodsCache();
		foreach (var loadedPackage in loadedPackagesSnapshot)
		foreach (var type in loadedPackage.Types.Values.ToArray())
			type.ReimplementGenericTypeMethods();
	}

	private readonly List<Package> loadedPackages = [];

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

	public const string DevelopmentBaseFolder = @"C:\code\GitHub\";
	internal static string CacheFolder =>
		Path.Combine( //ncrunch: no coverage, only downloaded and cached on non-development machines
			Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), StrictPackages);
	private const string StrictPackages = nameof(StrictPackages);
	public const string StrictOrg = "strict-lang";
	public static readonly Uri GitHubStrictUri = new("https://github.com/" + StrictOrg + "/");

	/// <summary>
	/// Called by Package.Dispose
	/// </summary>
	internal void Remove(Package result)
	{
		cacheService.Remove(result.FullName);
		lock (loadedPackages)
			loadedPackages.Remove(result);
	}

	public bool ContainsPackageNameInCache(string fullName) =>
		cacheService.TryGetValue<AsyncLazy<Package>>(fullName, out _);

	public async Task<string> ToDebugString() =>
		nameof(Repositories) +
		"\nStrict: " + (cacheService.TryGetValue<AsyncLazy<Package>>(nameof(Strict),
			out var lazyPackage)
			? (await lazyPackage.Value).ToDebugString()
			: "") +
		// ReSharper disable once InconsistentlySynchronizedField
		"\nLoadedPackages: " + string.Join("\n  ", loadedPackages) +
		"\nPreviouslyCheckedDirectories: " + string.Join<string>(", ", PreviouslyCheckedDirectories.ToList());
}