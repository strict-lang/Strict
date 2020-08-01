namespace Strict.Language
{
	public abstract class NamedType
	{
		protected NamedType(Context definedIn, string nameAndType)
		{
			var parts = nameAndType.Split(' ');
			Name = parts[0];
			if (!Name.IsWord())
				throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
			Type = definedIn.GetType(parts.Length == 1
				? parts[0].MakeFirstLetterUppercase()
				: parts[1]);
		}

		public string Name { get; }
		public Type Type { get; }
		public override string ToString() => Name + " " + Type;
	}
}