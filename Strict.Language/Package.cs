namespace Strict.Language
{
	//TODO: newest idea is to merge Type, TypeParser and TypeContext all in one class, maybe leave the parsing outside, but each Type could just be a context?

	/// <summary>
	/// In C# or Java called namespace or package as well, in Strict this is any code folder.
	/// </summary>
	public class Package : Context
	{
		public Package(string packageName) : base(packageName) { }

		/// <summary>
		/// Contains all high level <see cref="Package"/>. It itself is empty, has no parent and
		/// just contains all root children packages.
		/// </summary>
		public class Root : Context
		{
			public Root() : base(string.Empty) { }
		}

		public Package(Package parentPackage, string folderName) : base(folderName) { }
	}
	/*simply Type
	public class TypeContext : Context
	{
		public Context(Package package, Type type){}

	}
	*/
	/*simply Method
	public class MethodContext
	{
		public Context(TypeContext type, Method method){}

	}
	*/
	/// <summary>
	/// Keeps all known types for use, if in <see cref="Package"/> contains all known types
	/// and traits the context is inside a type, all members are available as
	/// well, in a method more scope information is available. The high level context knows it all.
	/// </summary>
	public abstract class Context
	{
		protected Context(string name) => Name = name;
		public string Name { get; }

		public Type GetType(string name)
		{
			if (name == Name)
				return (Type)this;
			return FindType(name)!;
		}
		
		public Type? FindType(string name)
		{
			return null;

		}
		/*not needed?
		public Type ParentType { get; } = Type.None;

		public Trait GetTrait(string word)
		{
			return null!;
		}
		*/
	}
}