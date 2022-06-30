using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class List : Expression
{
	public List(Context context, List<Expression> values) : base(context.GetType(Base.List)) => Values = values;
	public List<Expression> Values { get; }
	public override string ToString() => "(" + ValuesToString() + ")";

	private string ValuesToString() =>
		Values.Aggregate("", (current, expression) => current + (expression + ", ")).Trim()[..^1];

	public static Expression? TryParse(Method.Line line, string input) =>
		input.Length > 2 && input[0] == '(' && input[^1] == ')'
			? TryParseList(line, input[1..^1].Split(","))
			: null;

	private static Expression TryParseList(Method.Line line, IEnumerable<string> elements) =>
		new List(line.Method,
			elements.Select(element => line.Method.TryParseExpression(line, element.Trim()) ??
				throw new MethodExpressionParser.UnknownExpression(line, element)).ToList());

	public static bool HasIncompatibleDimensions(Expression left, Expression right) =>
		left is List leftList && right is List rightList &&
		leftList.Values.Count != rightList.Values.Count;

	//TODO: Need some alternate approach like extension method to compare types easily
	public static bool HasMismatchingTypes(Expression left, Expression right) =>
		left is List leftList && !leftList.IsFirstType<Text>() && right switch
		{
			List rightList when rightList.IsFirstType<Text>() => true,
			Binary { Left: List rightBinaryLeftList } when rightBinaryLeftList.IsFirstType<Text>() =>
				true,
			Binary { Left: Text } => true,
			_ => !leftList.IsFirstType<Text>() && right is Text
		};

	private bool IsFirstType<T>() => Values.First() is T;

	public sealed class ListsHaveDifferentDimensions : ParsingFailed
	{
		public ListsHaveDifferentDimensions(Method.Line line, string error) : base(line, error) { }
	}
}