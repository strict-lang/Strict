using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <inheritdoc />
/// <summary>
/// In C# or Java called namespace or package as well, in Strict this is any code folder.
/// </summary>
public class Package : Context
{
	public Package(string packagePath) : this(RootForPackages, packagePath) { }

	// ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable, unique
	private static readonly Root RootForPackages = new();

	/// <summary>
	/// Contains all high level <see cref="Package"/>. Just contains the fallback None type (think
	/// void) and Boolean, has no parent and just contains all root children packages. Also features
	/// a cache of types searched from here so future access is much faster. See green comment here:
	/// https://strict.dev/img/FindType2020-07-01.png
	/// </summary>
	private sealed class Root : Package
	{
		public Root() : base(null!, string.Empty)
		{
			none = new Type(this, Base.None, null!);
			boolean = new Type(this, Base.Boolean, null!);
		}

		private readonly Type none;
		private readonly Type boolean;

		// ReSharper disable once MethodTooLong
		public override Type? FindType(string name, Context? searchingFrom = null)
		{
			if (name == Base.None)
				return none;
			if (name == Base.Boolean)
				return boolean;
			if (name == lastName)
				return lastType;
			if (cachedFoundTypes.TryGetValue(name, out var previouslyFoundType))
				return previouslyFoundType; //ncrunch: no coverage
			var type = FindTypeInChildrenPackages(name, searchingFrom as Package);
			if (type == null)
				return null;
			lastName = name;
			lastType = type;
			cachedFoundTypes.Add(name, type);
			return type;
		}

		private string lastName = "";
		private Type lastType = null!;
		private readonly Dictionary<string, Type> cachedFoundTypes = new();
	}

	public Package(Package parentPackage, string packagePath) : base(parentPackage,
		Path.GetFileName(packagePath))
	{
		FolderPath = packagePath;
		// ReSharper disable once ConstantConditionalAccessQualifier, needed for Root package
		parentPackage?.children.Add(this);
	}

	public string FolderPath { get; }
	private readonly List<Package> children = new();
	internal void Add(Type type) => types.Add(type);
	private readonly List<Type> types = new();

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
	public override Type? FindType(string name, Context? searchingFrom = null) =>
		FindDirectType(name) ?? (IsPrivateName(name)
			? null
			: (children.Count > 0
				? FindTypeInChildrenPackages(name, searchingFrom ?? this)
				: null) ?? Parent.FindType(name, this));

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Type? FindDirectType(string name)
	{
		foreach (var t in types)
			if (t.Name == name)
				return t;
		return null;
	}

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

	public Package? FindSubPackage(string name) =>
		children.FirstOrDefault(p => p.Name == name || p.ToString() == name);

	public Package? Find(string name) =>
		FindSubPackage(name) ?? RootForPackages.FindSubPackage(name);
}