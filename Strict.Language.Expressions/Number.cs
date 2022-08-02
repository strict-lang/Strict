using System;
using System.Globalization;

namespace Strict.Language.Expressions;

public sealed class Number : Value
{
	public Number(Context context, double value) : base(context.GetType(Base.Number), value) { }
	public override string ToString() => ((double)Data).ToString(CultureInfo.InvariantCulture);

	public override bool Equals(Expression? other) =>
		other is Value v && (double)Data == (double)v.Data;

	public static Expression? TryParse(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		return input.Length < 1
			? null
			: int.TryParse(input, out var number)
				? new Number(line.Method, number)
				: double.TryParse(input, out var doubleNumber)
					? new Number(line.Method, doubleNumber)
					: null;
	}
}

//https://deltaengine.fogbugz.com/f/cases/25307