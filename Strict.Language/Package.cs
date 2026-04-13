using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Transpiler.Cuda")]

namespace Strict.Language;

/// <summary>
/// In C# or Java called namespace or package as well, in Strict this is any code folder.
/// </summary>
public class Package : Context, IDisposable
{
#if DEBUG
	public Package(string packagePath, Repositories? createdFromRepos = null,
		[CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = "") : this(RootForPackages, packagePath,
		createdFromRepos, callerFilePath, callerLineNumber, callerMemberName) { }
#else
	public Package(string packagePath, Repositories? createdFromRepos = null)
		: this(RootForPackages, packagePath, createdFromRepos) { }
#endif
#if DEBUG
	public Package(Package? parentPackage, string packagePath, Repositories? createdFromRepos = null,
		[CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = "") : base(parentPackage,
		Path.GetFileName(packagePath), callerFilePath, callerLineNumber, callerMemberName)
#else
	public Package(Package? parentPackage, string packagePath, Repositories? createdFromRepos = null)
		: base(parentPackage, Path.GetFileName(packagePath))
#endif
	{
		this.createdFromRepos = createdFromRepos;
		FolderPath = Path.IsPathRooted(packagePath)
			? packagePath
			: Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg,
				packagePath.Replace("TestPackage", "Strict"));
		if (parentPackage == null)
			return;
		lock (parentPackage.syncRoot)
		{
			var existing = parentPackage.children.FirstOrDefault(existing => existing.Name == Name);
			if (existing != null)
				return;//TODO? throw new PackageAlreadyExists(Name, parentPackage, existing); //ncrunch: no coverage
			parentPackage.children.Add(this);
		}
	}

	public class PackageAlreadyExists(string name, Package parentPackage, Package existing)
		: Exception(name + " in " + (parentPackage.Name == "" //ncrunch: no coverage
				? nameof(Root)
				: "parent package " + parentPackage) + ", existing package " + existing.Name
#if DEBUG
			+ ", existing package created by " + existing.callerFilePath + ":" +
			existing.callerLineNumber + " from method " + existing.callerMemberName
#endif
		);

	private static readonly Root RootForPackages = new();
	private readonly Repositories? createdFromRepos;
	public string FolderPath { get; }

	/// <summary>
	/// Contains all high level <see cref="Package"/>s. Just contains the fallback None type (think
	/// void), has no parent, and just contains all root children packages. Also features a cache of
	/// types searched from here so future access is much faster. See the green comment here:
	/// https://strict-lang.org/img/FindType2020-07-01.png
	/// </summary>
	private sealed class Root : Package
	{
#if DEBUG
		public Root([CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
			[CallerMemberName] string callerMemberName = "") : base(null, string.Empty, null,
			callerFilePath, callerLineNumber, callerMemberName) =>
#else
		public Root() : base(null, string.Empty) =>
#endif
			cachedFoundTypes.Add(Type.None, new Type(this, new TypeLines(Type.None)));

		public override Type? FindTypeCore(string name, Context? searchingFrom = null) =>
			TryGetCachedType(name, out var previouslyFoundType)
				? previouslyFoundType
				: FindTypeInChildrenAndCache(name, searchingFrom);

		private bool TryGetCachedType(string name, out Type? type)
		{
			lock (syncRoot)
				return cachedFoundTypes.TryGetValue(name, out type);
		}

		private Type? FindTypeInChildrenAndCache(string name, Context? searchingFrom)
		{
			var type = FindTypeInChildrenPackages(name, searchingFrom as Package);
			if (type != null)
				cachedFoundTypes[name] = type;
			return type;
		}

		private readonly Dictionary<string, Type> cachedFoundTypes = new(StringComparer.Ordinal);
	}

	private readonly List<Package> children = new();
	private readonly Lock syncRoot = new();

	internal void Add(Type type)
	{
		lock (syncRoot)
			types.Add(type.Name, type);
	}

	private readonly Dictionary<string, Type> types = new();

	public Type? FindFullType(string fullName)
	{
		if (fullName.Contains(' ') || fullName.Contains('"'))
			return null; //ncrunch: no coverage
		var parts = fullName.Split(Context.ParentSeparator);
		if (parts.Length < 2)
			throw new FullNameMustContainPackageAndTypeNames(fullName);
		if (IsPrivateName(parts[^1]))
			throw new PrivateTypesAreOnlyAvailableInItsPackage(fullName);
		if (!fullName.StartsWith(FullName + Context.ParentSeparator, StringComparison.Ordinal))
			return (Parent as Package)?.FindFullType(fullName);
		var subName = fullName.Replace(FullName + Context.ParentSeparator, "");
		return subName.Contains(Context.ParentSeparator)
			? FindSubPackage(subName.Split(Context.ParentSeparator)[0])?.FindFullType(fullName)
			: FindDirectType(subName);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsPrivateName(string name) => char.IsLower(name[0]);

	public sealed class FullNameMustContainPackageAndTypeNames(string fullName) : Exception(fullName);

	public sealed class PrivateTypesAreOnlyAvailableInItsPackage(string fullName)
		: Exception(fullName);

	/// <summary>
	/// The following picture shows the typical search steps and optimizations done. It is different
	/// from simple binary searches or finding types in other languages because in Strict any public
	/// type can be used at any place. https://strict-lang.org/img/FindType2020-07-01.png
	/// </summary>
	public override Type? FindTypeCore(string name, Context? searchingFrom = null)
	{
		lock (syncRoot)
			if (name == lastName && lastType != null)
				return lastType;
		if (IsPrivateName(name))
			return null;
		var type = FindDirectType(name) ?? FindTypeInChildrenOrParentPackages(name, searchingFrom);
		if (type != null)
			lock (syncRoot)
			{
				lastName = name;
				lastType = type;
			}
		return type;
	}

	private Type? FindTypeInChildrenOrParentPackages(string name, Context? searchingFrom)
	{
		Type? type = null;
		lock (syncRoot)
			if (children.Count > 0)
				type = FindTypeInChildrenPackages(name, searchingFrom ?? this);
		type ??= Parent?.FindTypeCore(name, this);
		return type;
	}

	private string lastName = "";
	private Type? lastType;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Type? FindDirectType(string name)
	{
		lock (syncRoot)
			return types.GetValueOrDefault(name);
	}

	private Type? FindTypeInChildrenPackages(string name, Context? searchingFromPackage)
	{
		Package[] childrenSnapshot;
		lock (syncRoot)
			childrenSnapshot = children.ToArray();
		foreach (var t in childrenSnapshot)
			if (t != searchingFromPackage)
			{
				var childType = t.FindDirectType(name) ?? (childrenSnapshot.Length > 0
					? t.FindTypeInChildrenPackages(name, searchingFromPackage)
					: null);
				if (childType != null)
					return childType;
			}
		return null;
	}

	public Package? FindSubPackage(string name)
	{
		Package[] childrenSnapshot;
		lock (syncRoot)
			childrenSnapshot = children.ToArray();
		foreach (var child in childrenSnapshot)
			if (child.Name == name || child.FullName == name)
				return child;
		return null;
	}

	public Package? Find(string name) =>
		FindSubPackage(name) ?? RootForPackages.FindSubPackage(name);

	public void Remove(Type? type)
	{
		if (type != null)
			lock (syncRoot)
				types.Remove(type.Name);
	}

	internal void Remove(Package package)
	{
		lock (syncRoot)
			children.Remove(package);
	}

	public IReadOnlyDictionary<string, Type> Types => types;
	internal List<Package> automaticallyLoadedDependencyPackages = new();

	public void Dispose()
	{
		/*
		GC.SuppressFinalize(this);
		while (children.Count > 0)
			children[0].Dispose();
		foreach (var dependencyPackage in automaticallyLoadedDependencyPackages)
			dependencyPackage.Dispose();
		foreach (var type in types)
			type.Value.Dispose();
		//Console.WriteLine("Package.Dispose " + FullName);
		// ReSharper disable once ConditionalAccessQualifierIsNonNullableAccordingToAPIContract
		((Package)Parent)?.Remove(this);
		createdFromRepos?.Remove(this);
		*/
	}
}