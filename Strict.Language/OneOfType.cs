﻿namespace Strict.Language;

public sealed class OneOfType(Type definedInType, IReadOnlyList<Type> types) : Type(
	definedInType.Package,
	new TypeLines(string.Join("Or", types.Select(t => t.Name)), GetOneOfTypeLines(types)))
{
	private static string[] GetOneOfTypeLines(IReadOnlyList<Type> types)
	{
		var lines = new string[types.Count];
		for (var index = 0; index < types.Count; index++)
			lines[index] = HasWithSpaceAtEnd + types[index].Name;
		return lines;
	}

	public IReadOnlyList<Type> Types { get; } = types;
}