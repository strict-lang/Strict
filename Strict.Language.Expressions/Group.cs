using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Strict.Language.Expressions;

public class Group
{
	public static Expression? TryParse(Method.Line line, string input)
	{
		var findGroupsRegex = new Regex(@"\([a-zA-Z0-9]+(\s+(\+|\-|\*|\/|is)\s[a-zA-Z0-9]+)+\)");
		var matchedGroups = findGroupsRegex.Matches(input);
		if (matchedGroups.Count > 0)
		{
			var expressions = new List<Expression?>();
			foreach (Match group in matchedGroups)
			{
				expressions.Add(Binary.TryParse(line, group.Value.Replace("(", "").Replace(")", "")));
			}
			return expressions.First(); // need all expressions
		}
		return null;
	}
}