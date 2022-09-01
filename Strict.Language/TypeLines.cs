using System;
using System.Collections.Generic;

namespace Strict.Language;

/// <summary>
/// Optimization to split type parsing into three steps:
/// Step 1 is to just load the file lines (or mock it) and inspect the implement lines only.
/// Step 2 is to sort by dependencies and create each of the types
/// Step 3 is to actually resolve each of the types members and members (which can be in a
/// different order as well, methods are also evaluated lazily and not at parsing time)
/// </summary>
public class TypeLines
{
	public TypeLines(string name, params string[] lines)
	{
		Name = name;
		Lines = lines;
		ImplementTypes = ExtractImplementTypes();
	}

	public string Name { get; }
	public string[] Lines { get; }
	public IReadOnlyList<string> ImplementTypes { get; }

	private IReadOnlyList<string> ExtractImplementTypes()
	{
		IList<string> implements = Array.Empty<string>();
		foreach (var line in Lines)
			if (line.StartsWith(Type.Implement, StringComparison.Ordinal))
			{
				if (implements.Count == 0)
					implements = new List<string>();
				AddImplements(line, implements);
			}
			else
				break;
		return (IReadOnlyList<string>)implements;
	}

	private static void AddImplements(string line, ICollection<string> implements)
	{
		var remainingLine = line[Type.Implement.Length..];
		if (remainingLine.Contains('('))
			foreach (var part in remainingLine.Split(new[] { '(', ')', ',' },
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
				implements.Add(part);
		else
			implements.Add(remainingLine);
	}

	public override string ToString() => Name + ImplementTypes.ToBrackets();
}