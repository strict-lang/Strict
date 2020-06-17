namespace Strict.Language
{
	public abstract class NamedType
	{
		protected NamedType(string name, Type type)
		{
			Name = name;
			Type = type;
		}

		public string Name { get; }
		public Type Type { get; }
	}
}