using System;
using System.Runtime.CompilerServices;

namespace Strict.Language.Expressions;

public sealed class Boolean : Value
{
	public Boolean(Context context, bool value) : base(context.GetType(Base.Boolean), value) { }
	public override string ToString() => base.ToString().ToLower();

	public override bool Equals(Expression? other) =>
		other is Value v && (bool)Data == (bool)v.Data;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static Expression? TryParse(Method.Line line, Range partToParse)
	{
		var part = line.Text.GetSpanFromRange(partToParse);
		return part.IsTrueText()
			? new Boolean(line.Method, true)
			: part.IsFalseText()
				? new Boolean(line.Method, false)
				: null;
	}
}