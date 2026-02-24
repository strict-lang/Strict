using System.Collections;
using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <summary>
/// In C# or Java called namespace or package as well, in Strict this is any code folder.
/// </summary>
public class Package : Context, IEnumerable<Type>, IDisposable
{
#if DEBUG
	public Package(string packagePath, [CallerFilePath] string callerFilePath = "",
		[CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = "") : this(RootForPackages, packagePath,
		// ReSharper disable ExplicitCallerInfoArgument
		callerFilePath, callerLineNumber, callerMemberName) { }
#else
	public Package(string packagePath) : this(RootForPackages, packagePath) { }
#endif
	private static readonly Root RootForPackages = new();

	/// <summary>
	/// Contains all high level <see cref="Package"/>s. Just contains the fallback None type (think
	/// void), has no parent and just contains all root children packages. Also features a cache of
	/// types searched from here so future access is much faster. See green comment here:
	/// https://strict.dev/img/FindType2020-07-01.png
	/// </summary>
	private sealed class Root : Package
	{
#if DEBUG
		public Root([CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
			[CallerMemberName] string callerMemberName = "") : base(null, string.Empty, callerFilePath,
			callerLineNumber, callerMemberName) =>
#else
		public Root() : base(null, string.Empty) =>
#endif
			cachedFoundTypes.Add(Base.None, new Type(this, new TypeLines(Base.None)));

		public override Type? FindType(string name, Context? searchingFrom = null) =>
			cachedFoundTypes.TryGetValue(name, out var previouslyFoundType)
				? previouslyFoundType
				: FindTypeInChildrenAndCache(name, searchingFrom);

		private Type? FindTypeInChildrenAndCache(string name, Context? searchingFrom)
		{
			var type = FindTypeInChildrenPackages(name, searchingFrom as Package);
			cachedFoundTypes.Add(name, type!);
			return type;
		}

		private readonly Dictionary<string, Type> cachedFoundTypes = new(StringComparer.Ordinal);
	}

#if DEBUG
	public Package(Package? parentPackage, string packagePath,
		[CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = "") : base(parentPackage,
		Path.GetFileName(packagePath), callerFilePath, callerLineNumber, callerMemberName)
#else
	public Package(Package? parentPackage, string packagePath)
		: base(parentPackage, Path.GetFileName(packagePath))
#endif
	{
		FolderPath = packagePath;
		if (parentPackage == null)
			return;
		var existing = parentPackage.children.FirstOrDefault(existingPackage => existingPackage.Name == Name);
		if (existing != null)
			throw new PackageAlreadyExists(Name, parentPackage, existing); //ncrunch: no coverage
		parentPackage.children.Add(this);
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

	public string FolderPath { get; }
	private readonly List<Package> children = new();
	internal void Add(Type type) => types.Add(type.Name, type);
	private readonly Dictionary<string, Type> types = new();

	public Type? FindFullType(string fullName)
	{
		var parts = fullName.Split('.');
		if (parts.Length < 2)
			throw new FullNameMustContainPackageAndTypeNames();
		if (IsPrivateName(parts[^1]))
			throw new PrivateTypesAreOnlyAvailableInItsPackage();
		if (!fullName.StartsWith(ToString() + ".", StringComparison.Ordinal))
			return (Parent as Package)?.FindFullType(fullName);
		var subName = fullName.Replace(ToString() + ".", "");
		return subName.Contains('.')
			? FindSubPackage(subName.Split('.')[0])?.FindFullType(fullName)
			: FindDirectType(subName);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsPrivateName(string name) => char.IsLower(name[0]);

	public sealed class FullNameMustContainPackageAndTypeNames : Exception;
	public sealed class PrivateTypesAreOnlyAvailableInItsPackage : Exception;

	/// <summary>
	/// The following picture shows the typical search steps and optimizations done. It is different
	/// from simple binary searchs or finding types in other languages because in Strict any public
	/// type can be used at any place. https://strict.dev/img/FindType2020-07-01.png
	/// </summary>
	public override Type? FindType(string name, Context? searchingFrom = null)
	{
		if (name == lastName)
			return lastType;
		if (IsPrivateName(name))
			return null;
		var type = FindDirectType(name) ??
			FindTypeInChildrenOrParentPackages(name, searchingFrom);
		lastName = name;
		lastType = type;
		return type;
	}

	private Type? FindTypeInChildrenOrParentPackages(string name, Context? searchingFrom)
	{
		Type? type = null;
		if (children.Count > 0)
			type = FindTypeInChildrenPackages(name, searchingFrom ?? this);
		type ??= Parent.FindType(name, this);
		return type;
	}

	private string lastName = "";
	private Type? lastType;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Type? FindDirectType(string name) => types.GetValueOrDefault(name);

	private Type? FindTypeInChildrenPackages(string name, Context? searchingFromPackage)
	{
		foreach (var t in children)
			if (t != searchingFromPackage)
			{
				var childType = t.FindDirectType(name) ?? (children.Count > 0
					? t.FindTypeInChildrenPackages(name, searchingFromPackage)
					: null);
				if (childType != null)
					return childType;
			}
		return null;
	}

	public Package? FindSubPackage(string name)
	{
		foreach (var child in children)
			if (child.Name == name || child.FullName == name)
				return child;
		return null;
	}

	public Package? Find(string name) =>
		FindSubPackage(name) ?? RootForPackages.FindSubPackage(name);

	public void Remove(Type? type)
	{
		if (type != null)
			types.Remove(type.Name);
	}

	internal void Remove(Package package) => children.Remove(package);
	public IEnumerator<Type> GetEnumerator() => new List<Type>(types.Values).GetEnumerator();
	IEnumerator IEnumerable.GetEnumerator() => GetEnumerator(); //ncrunch: no coverage

	// ReSharper disable once ConditionalAccessQualifierIsNonNullableAccordingToAPIContract
	public void Dispose() => ((Package)Parent)?.Remove(this);
}