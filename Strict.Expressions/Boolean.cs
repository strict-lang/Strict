using System.Runtime.CompilerServices;
using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// Constant boolean that appears anywhere in the parsed code, simply "true" or "false"
/// </summary>
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