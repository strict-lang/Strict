using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Text(Context context, string value, int lineNumber = 0)
	: Value(context.GetType(Type.Text), value, lineNumber)
{
	/// <summary>
	/// Text must start and end with a double quote, only called for input that does not contain any
	/// spaces, or if there are spaces (like "Hello World"), they are already split via
	/// <see cref="PhraseTokenizer"/> and thus is only a string.
	/// </summary>
	public static Expression? TryParse(Body body, ReadOnlySpan<char> input) =>
		input.Length >= 2 && input[0] == '"' && input[^1] == '"'
			? new Text(body.Method, Unescape(input.Slice(1, input.Length - 2)),
				body.CurrentFileLineNumber)
			: null;

	private static string Unescape(ReadOnlySpan<char> input) =>
		input.ToString().Replace("\\n", "\n", StringComparison.Ordinal).
			Replace("\\r", "\r", StringComparison.Ordinal).
			Replace("\\t", "\t", StringComparison.Ordinal).
			Replace("\\\"", "\"", StringComparison.Ordinal).
			Replace(@"\\", @"\", StringComparison.Ordinal);
}