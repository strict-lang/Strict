namespace Strict.Language;

/// <summary>
/// Optimization to split type parsing into three steps:
/// Step 1 is to just load the file lines (or mock it) and inspect the member lines only.
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
		DependentTypes = ExtractDependentTypes();
	}

	public string Name { get; }
	public string[] Lines { get; }
	public IReadOnlyList<string> DependentTypes { get; }

	private IReadOnlyList<string> ExtractDependentTypes()
	{
		// Often there are no members, no need to create a new empty list
		IList<string> dependentTypes = Array.Empty<string>();
		foreach (var line in Lines)
			if (line.StartsWith(Type.HasWithSpaceAtEnd, StringComparison.Ordinal))
				AddDependentType(line[Type.HasWithSpaceAtEnd.Length..], ref dependentTypes);
			else if (line.StartsWith(Type.MutableWithSpaceAtEnd, StringComparison.Ordinal))
				AddDependentType(line[Type.MutableWithSpaceAtEnd.Length..], ref dependentTypes);
			else if (!line.StartsWith('\t'))
				AddDependentTypesFromMethodParametersAndReturnType(line, ref dependentTypes);
			else
				break;
		return (IReadOnlyList<string>)dependentTypes;
	}

	private void AddDependentTypesFromMethodParametersAndReturnType(string line,
		ref IList<string> dependentTypes)
	{
		var startIndex = line.Contains('(')
			? line.IndexOf('(')
			: line.IndexOf(' ');
		if (startIndex > 0)
			AddDependentType(line[startIndex..], ref dependentTypes);
	}

	private void AddDependentType(string remainingLine, ref IList<string> dependentTypes)
	{
		if (dependentTypes.Count == 0)
			dependentTypes = new List<string>();
		if (remainingLine.Contains('('))
			foreach (var part in remainingLine.Split(new[] { '(', ')', ',' },
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
				AddIfNotExisting(dependentTypes, part);
		else if (remainingLine.EndsWith('s'))
		{
			AddIfNotExisting(dependentTypes, Base.List);
			AddIfNotExisting(dependentTypes, remainingLine[..^1].MakeFirstLetterUppercase());
		}
		else
			AddIfNotExisting(dependentTypes, remainingLine.MakeFirstLetterUppercase());
	}

	private void AddIfNotExisting(ICollection<string> dependentTypes, string typeName)
	{
		if (typeName.Contains(Keyword.With))
			typeName = typeName[..typeName.IndexOf("with", StringComparison.Ordinal)].Trim();
		else if (typeName.Contains(' '))
			typeName = typeName.Split(' ')[1];
		if (!dependentTypes.Contains(typeName.MakeFirstLetterUppercase()) && Name != typeName &&
			!typeName.IsKeyword() && typeName != Base.Generic)
			dependentTypes.Add(typeName.MakeFirstLetterUppercase());
	}

	public override string ToString() => Name + DependentTypes.ToBrackets();
}