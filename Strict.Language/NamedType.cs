namespace Strict.Language
{
	public abstract class NamedType
	{
		protected NamedType(string name, Type type)
		{
			Name = name;
			Type = type;
		}

		public string Name { get; }//TODO: regex: ([a-zA-Z]+[a-zA-Z0-9_]*), not sure if we even should allow numbers, normally not needed!
		public Type Type { get; } //([\w<>]+)

		public override string ToString() => Name + " " + Type;
	}
}