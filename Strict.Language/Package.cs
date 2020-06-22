using System.Collections.Generic;

namespace Strict.Language
{
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
		public class Root : Package
		{
			public Root() : base(null!, string.Empty)
			{
				new Type(this, Base.None, "");
				new Type(this, Base.Boolean, "");
			}

			public override Type? FindType(string name, Type? searchingFromType = null) =>
				name == Base.None
					? types[0]
					: name == Base.Boolean
						? types[1]
						: null;
		}

		public Package(Package parentPackage, string folderName) : base(parentPackage, folderName) { }

		internal void Add(Type type) => types.Add(type);
		private readonly List<Type> types = new List<Type>();

		public override Type? FindType(string name, Type? searchingFromType = null) =>
			types.Find(t => t.Name == name) ?? types.Find(t => t.FullName == name) ??
			Parent.FindType(name) ?? FindTypeInChildren(name, searchingFromType);

		private Type? FindTypeInChildren(string name, Type? searchingFromType)
		{
			foreach (var child in Children)
				if (child != searchingFromType)
				{
					var childType = child is Package
						? child.FindType(name, searchingFromType)
						: child.Name == name || child.FullName == name
							? child
							: null;
					if (childType != null)
						return (Type)childType;
				}
			return null;
		}

		public Type? FindDirectType(string name) => types.Find(t => t.Name == name);
	}
}