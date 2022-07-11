using System;
using System.Collections;
using System.Collections.Generic;

namespace Strict.Language;

public sealed class GroupParser
{
	public GroupParser(string code) => Groups = TryParse(code);
	public List<Group> Groups { get; }

	private static List<Group> TryParse(string code)
	{
		var groups = new List<Group>();
		var groupStacks = new Stack<Group>();
		for (var index = 0; index < code.Length; index++)
		{
			var isLastIndex = index == code.Length - 1;
			CheckUnbalancedBracketsInFront(code, isLastIndex, index);
			if (code[index] == '(')
				groupStacks.Push(new Group(index + 1));
			if (code[index] != ')')
				continue;
			AddGroupsFromStack(groupStacks, index, groups);
			CheckUnbalancedBracketsInEnd(isLastIndex, groupStacks, index);
		}
		return groups;
	}

	private static void AddGroupsFromStack(Stack<Group> groupStacks, int index, ICollection<Group> groups)
	{
		if (groupStacks.Count == 0)
			throw new UnbalancedBracketsFound(index);
		var currentGroup = groupStacks.Pop();
		currentGroup.Length = index - currentGroup.Start;
		groups.Add(currentGroup);
	}

	private static void CheckUnbalancedBracketsInEnd(bool isLastIndex, ICollection groupStacks, int index)
	{
		if (isLastIndex && groupStacks.Count > 0)
			throw new UnbalancedBracketsFound(index);
	}

	private static void CheckUnbalancedBracketsInFront(string code, bool isLastIndex, int index)
	{
		if (isLastIndex && code[index] == '(')
			throw new UnbalancedBracketsFound(index);
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