using System;
using System.Runtime.CompilerServices;

namespace Strict.Language.Expressions;

public class Boolean : Value
{
	public Boolean(Context context, bool value) : base(context.GetType(Base.Boolean), value) { }
	public override string ToString() => base.ToString().ToLower();

	public override bool Equals(Expression? other) =>
		other is Value v && (bool)Data == (bool)v.Data;

	[Obsolete]
	public static Expression? TryParse(Method.Line line, string partToParse) =>
		partToParse switch
		{
			"true" => new Boolean(line.Method, true),
			"false" => new Boolean(line.Method, false),
			_ => null
		};

	[Obsolete]
	public static Expression? TryParse(Method.Line line, Tuple<int, int> startAndLength) =>
		line.Text.Substring(startAndLength.Item1, startAndLength.Item2) switch
		{
			"true" => new Boolean(line.Method, true),
			"false" => new Boolean(line.Method, false),
			_ => null
		};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static Expression? TryParse(Method.Line line, Range partToParse)
	{
		var part = line.Text.GetSpanFromRange(partToParse);
		return part.Compare("true".AsSpan())
			? new Boolean(line.Method, true)
			: part.Compare("false".AsSpan())
				? new Boolean(line.Method, false)
				: null;
	}
}