using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <inheritdoc />
/// <summary>
/// In C# or Java called namespace or package as well, in Strict this is any code folder.
/// </summary>
public class Package : Context
{
	public Package(string packagePath) : this(RootForPackages, packagePath) { }
	private static readonly Root RootForPackages = new();

	/// <summary>
	/// Contains all high level <see cref="Package"/>s. Just contains the fallback None type (think
	/// void), has no parent and just contains all root children packages. Also features a cache of
	/// types searched from here so future access is much faster. See green comment here:
	/// https://strict.dev/img/FindType2020-07-01.png
	/// </summary>
	private sealed class Root : Package
	{
		public Root() : base(null, string.Empty) =>
			cachedFoundTypes.Add(Base.None,
				new Type(this, new TypeLines(Base.None)));

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

	public Package(Package? parentPackage, string packagePath) : base(parentPackage,
		Path.GetFileName(packagePath))
	{
		FolderPath = packagePath;
		parentPackage?.children.Add(this);
	}

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

	public sealed class FullNameMustContainPackageAndTypeNames : Exception { }
	public sealed class PrivateTypesAreOnlyAvailableInItsPackage : Exception { }

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
	public Type? FindDirectType(string name) =>
		types.TryGetValue(name, out var type)
			? type
			: null;

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

	public void Remove(Type type) => types.Remove(type.Name);
}