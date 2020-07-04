using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Strict.Language
{
	/// <inheritdoc />
	/// <summary>
	/// In C# or Java called namespace or package as well, in Strict this is any code folder.
	/// </summary>
	public class Package : Context
	{
		public Package(string packagePath) :
			base(RootForPackages, Path.GetFileName(packagePath)) =>
			FolderPath = packagePath;

		// ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable, unique for all packages
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

		public Package(Package parentPackage, string packagePath) : base(parentPackage,
			Path.GetFileName(packagePath)) =>
			FolderPath = packagePath;

		public string FolderPath { get; set; }
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
	}
}