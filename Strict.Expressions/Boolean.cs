using System.Runtime.CompilerServices;
using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// Constant boolean that appears anywhere in the parsed code, simply "true" or "false"
/// </summary>
public sealed class Boolean(Context context, bool value, int lineNumber = 0)
	: Value(context.GetType(Base.Boolean), value, lineNumber)
{
	public override string ToString() => base.ToString().ToLower();

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.IsTrueText()
			? new Boolean(body.Method, true, body.CurrentFileLineNumber)
			: line.IsFalseText()
				? new Boolean(body.Method, false, body.CurrentFileLineNumber)
				: null;
}