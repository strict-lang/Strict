using System;
using System.Collections;
using System.Collections.Generic;

namespace Strict.Language;

public sealed class BracketParser
{
	public BracketParser(string code) => Groups = TryParse(code);
	public List<Group> Groups { get; }

	private static List<Group> TryParse(string code)
	{
		var groupStacks = new Stack<Group>();
		return ParseByCharacter(code, groupStacks);
	}

	private static List<Group> ParseByCharacter(string code, Stack<Group> groupStacks)
	{
		var groups = new List<Group>();
		for (var index = 0; index < code.Length; index++)
		{
			PushGroupToStack(code, index, groupStacks);
			if (code[index] != ')')
				continue;
			PopGroupFromStack(groupStacks, index, groups);
			CheckBracketsAtEnd(index == code.Length - 1, groupStacks, index);
		}
		return groups;
	}

	private static void PushGroupToStack(string code, int index, Stack<Group> groupStacks)
	{
		if (code[index] != '(')
			return;
		if (index == code.Length - 1)
			throw new UnbalancedBracketsFound(index);
		groupStacks.Push(new Group(index + 1));
	}

	private static void CheckBracketsAtEnd(bool isLastIndex, ICollection groupStacks, int index)
	{
		if (isLastIndex && groupStacks.Count > 0)
			throw new UnbalancedBracketsFound(index);
	}

	private static void PopGroupFromStack(Stack<Group> groupStacks, int index, ICollection<Group> groups)
	{
		if (groupStacks.Count == 0)
			throw new UnbalancedBracketsFound(index);
		var currentGroup = groupStacks.Pop();
		currentGroup.Length = index - currentGroup.Start;
		groups.Add(currentGroup);
	}

	public sealed class UnbalancedBracketsFound : Exception
	{
		public UnbalancedBracketsFound(int index) : base("Position " + index) { }
	}
}

public sealed record Group(int Start)
{
	public int Length { get; set; }
}