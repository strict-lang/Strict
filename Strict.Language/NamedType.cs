namespace Strict.Language
{
	public abstract class NamedType
	{
		protected NamedType(Context definedIn, string nameAndType)
		{
			var parts = nameAndType.Split(' ');
			Name = parts[0];
			Type = definedIn.GetType(parts.Length == 1
				? parts[0].Substring(0, 1).ToUpperInvariant() + parts[0].Substring(1)
				: parts[1]);
		}
		
		public string Name { get; }//limit to regex: ([a-zA-Z]+[a-zA-Z]*)
		public Type Type { get; } //([\w<>]+)
		public override string ToString() => Name + " " + Type;
	}
}