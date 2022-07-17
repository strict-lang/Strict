using System;
using System.Globalization;

namespace Strict.Language.Expressions;

public class Number : Value
{
	public Number(Context context, double value) : base(context.GetType(Base.Number), value) { }
	public override string ToString() => ((double)Data).ToString(CultureInfo.InvariantCulture);

	public override bool Equals(Expression? other) =>
		other is Value v && (double)Data == (double)v.Data;

	public static Expression? TryParse(Method.Line line, string partToParse) =>
		partToParse.Length >= 1 && double.TryParse(partToParse, out var number)
			? new Number(line.Method, number)
			: null;

	public static Expression? TryParse(Method method, ReadOnlySpan<char> partToParse) =>
		partToParse.Length >= 1 && double.TryParse(partToParse, out var number)
			? new Number(method, number)
			: null;
}