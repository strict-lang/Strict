using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Strict.Language.Expressions;

public class Number : Value
{
	public Number(Context context, double value) : base(context.GetType(Base.Number), value) { }
	public override string ToString() => ((double)Data).ToString(CultureInfo.InvariantCulture);

	public override bool Equals(Expression? other) =>
		other is Value v && (double)Data == (double)v.Data;

	[Obsolete]
	public static Expression? TryParse(Method.Line line, string partToParse) =>
		partToParse.Length >= 1 && double.TryParse(partToParse, out var number)
			? new Number(line.Method, number)
			: null;

	public static Expression? TryParse(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		return input.Length >= 1 && double.TryParse(input, out var number)
			? new Number(line.Method, number)
			: null;
	}
}

//TODO: we need Range, Slice, Mutable, Error and looping soon https://strict.dev/docs/Keywords#buildin-types