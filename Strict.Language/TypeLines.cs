using System;
using System.Collections.Generic;

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
		MemberTypes = ExtractMemberTypes();
	}

	public string Name { get; }
	public string[] Lines { get; }
	public IReadOnlyList<string> MemberTypes { get; }

	private IReadOnlyList<string> ExtractMemberTypes()
	{
		// Often there are no members, no need to create a new empty list
		IList<string> members = Array.Empty<string>();
		foreach (var line in Lines)
			if (line.StartsWith(Type.Has, StringComparison.Ordinal))
				AddMemberType(line[Type.Has.Length..], ref members);
			else if (line.StartsWith(Type.Mutable, StringComparison.Ordinal))
				AddMemberType(line[Type.Mutable.Length..], ref members);
			else
				break;
		return (IReadOnlyList<string>)members;
	}

	private static void AddMemberType(string remainingLine, ref IList<string> memberTypes)
	{
		if (memberTypes.Count == 0)
			memberTypes = new List<string>();
		if (remainingLine.Contains('('))
			foreach (var part in remainingLine.Split(new[] { '(', ')', ',' },
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
				memberTypes.Add(part);
		else if (remainingLine.EndsWith('s'))
		{
			memberTypes.Add(Base.List);
			memberTypes.Add(remainingLine[..^1].MakeFirstLetterUppercase());
		}
		else
			memberTypes.Add(remainingLine);
	}

	public override string ToString() => Name + MemberTypes.ToBrackets();
}