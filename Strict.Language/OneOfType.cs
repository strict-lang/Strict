namespace Strict.Language;

public sealed class OneOfType(Type definedInType, Type[] types, string combinedName) : Type(
	definedInType.Package, new TypeLines(combinedName, GetOneOfTypeLines(types)))
{
	private static string[] GetOneOfTypeLines(IReadOnlyList<Type> types)
	{
		var lines = new string[types.Count];
		for (var index = 0; index < types.Count; index++)
			lines[index] = HasWithSpaceAtEnd + types[index].Name;
		return lines;
	}

	public Type[] Types { get; } = types;
	public override bool IsBoolean => Types.Any(t => t.IsBoolean);
	public static string BuildName(Type[] types) => string.Join("Or", types.Select(t => t.Name));
}