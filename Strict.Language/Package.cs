using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Strict.Language
{
	/// <inheritdoc />
	/// <summary>
	/// In C# or Java called namespace or package as well, in Strict this is any code folder.
	/// </summary>
	public class Package : Context
	{
		public Package(string packageName) : base(RootForPackages, packageName) { }

		private static readonly Root RootForPackages = new Root();

		/// <summary>
		/// Contains all high level <see cref="Package"/>. Just contains the fallback None type (think
		/// void) and Boolean, has no parent and just contains all root children packages.
		/// </summary>
		private class Root : Package
		{
			public Root() : base(null!, string.Empty)
			{
				new Type(this, Base.None, "");
				new Type(this, Base.Boolean, "");
			}

			public override Type? FindType(string name, Package? searchingFromPackage = null,
				Type? searchingFromType = null) =>
				name == Base.None || name == Base.Boolean
					? base.FindType(name, searchingFromPackage, searchingFromType)
					: null;
		}

		public Package(Package parentPackage, string folderName) : base(parentPackage, folderName) { }

		internal void Add(Type type) => types.Add(type);
		private readonly List<Type> types = new List<Type>();

		public override Type? FindType(string name, Package? searchingFromPackage = null,
			Type? searchingFromType = null) =>
			types.Find(t => t.Name == name) ?? (name.Contains(".")
				? types.Find(t => t.ToString() == name)
				: null) ?? AbortIfTypeIsPrivate(name) ??
			Parent.FindType(name, this, searchingFromType) ??
			FindTypeInChildren(name, searchingFromPackage, searchingFromType);

		private static Type? AbortIfTypeIsPrivate(string name) =>
			char.IsLower(name.Split('.').Last()[0])
				? throw new PrivateTypesAreOnlyAvailableInItsPackage()
				: (Type?)null;

		public class PrivateTypesAreOnlyAvailableInItsPackage : Exception {}

		private Type? FindTypeInChildren(string name, Package? searchingFromPackage, Type? searchingFromType)
		{
			foreach (var child in Children)
				if (child != searchingFromType && child != searchingFromPackage)
				{
					var childType = child is Package
						? child.FindType(name, searchingFromPackage, searchingFromType)
						: child.Name == name || child.ToString() == name
							? child
							: null;
					if (childType != null)
						return (Type)childType;
				}
			return null;
		}

		public Type? FindDirectType(string name) => types.Find(t => t.Name == name);
		public Package GetSubPackage(string name) => (Package)Children.First(p => p.Name == name);

		/// <summary>
		/// Loads a package from disc (or later any link like github) like Strict for base types
		/// </summary>
		public static Package FromDisk(string packageName)
		{
			if (packageName != nameof(Strict))
				throw new OnlyStrictPackageIsAllowed();
			return FromDiskPath(BasePath + packageName);
		}

		public class OnlyStrictPackageIsAllowed : Exception { }

		private static Package FromDiskPath(string packagePath) =>
			CreatePackageFromFiles(packagePath, RootForPackages,
				Directory.GetFiles(packagePath, "*" + Type.Extension));

		/// <summary>
		/// Initially we need to create just empty types and then after they all have been created
		/// we will fill and load them, otherwise we could not use types within the package context.
		/// </summary>
		private static Package CreatePackageFromFiles(string packagePath, Package parent,
			string[] files)
		{
			if (parent != RootForPackages && files.Length == 0)
				return null!;
			var package = new Package(parent, Path.GetFileName(packagePath));
			foreach (var filePath in files)
				new Type(package, Path.GetFileNameWithoutExtension(filePath), string.Empty);
			Parallel.For(0, package.types.Count,
				async index => await package.types[index].ParseFile(files[index]));
			foreach (var directory in Directory.GetDirectories(packagePath))
				CreatePackageFromFiles(directory, package,
					Directory.GetFiles(directory, "*" + Type.Extension));
			return package;
		}
		
		private static string BasePath => @"C:\code\GitHub\strict-lang\";
		public string LocalPath =>
			Path.Combine(BasePath, ToString().Replace('.', Path.DirectorySeparatorChar));
	}
}