﻿using System.Globalization;
using Strict.Language;

namespace Strict.Expressions;

public sealed class Number(Context context, double value)
	: Value(context.GetType(Base.Number), value)
{
	public override string ToString() => ((double)Data).ToString(CultureInfo.InvariantCulture);

	public override bool Equals(Expression? other) =>
		other is Value v && (double)Data == (double)v.Data;

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.TryParseNumber(out var number)
			? new Number(body.Method, number)
			: null;
}