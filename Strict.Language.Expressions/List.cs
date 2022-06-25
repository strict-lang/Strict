using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class List : Value
{
	public List<Expression> Values { get; }
	public List(Context context, List<Expression> values) : base(context.GetType(Base.List), values) => Values = values;
	public override string ToString() => "(" + ValuesToString() + ")";

	private string ValuesToString() =>
		Values.Aggregate("", (current, expression) => current + (expression + ", ")).Trim()[..^1];

	public static Expression? TryParse(Method.Line line, string input) =>
		input.Length > 2 && input[0] == '(' && input[^1] == ')'
			? TryParseList(line, input[1..^1].Split(","))
			: null;

	private static Expression TryParseList(Method.Line line, IEnumerable<string> elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
		{
			var foundExpression = line.Method.TryParseExpression(line, element.Trim());
			if (foundExpression != null)
				expressions.Add(foundExpression);
		}
		return new List(line.Method, expressions);
	}
}