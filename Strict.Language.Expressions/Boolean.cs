using System.Runtime.CompilerServices;

namespace Strict.Language.Expressions;

public sealed class Boolean(Context context, bool value)
	: Value(context.GetType(Base.Boolean), value)
{
	public override string ToString() => base.ToString().ToLower();

	public override bool Equals(Expression? other) =>
		other is Value v && (bool)Data == (bool)v.Data;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.IsTrueText()
			? new Boolean(body.Method, true)
			: line.IsFalseText()
				? new Boolean(body.Method, false)
				: null;
}