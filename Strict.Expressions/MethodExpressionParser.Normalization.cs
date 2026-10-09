using System.Runtime.CompilerServices;
using System.Text;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public partial class MethodExpressionParser
{
	//TODO: this is a hack and should be removed! we really want same input = output!
	private static string NormalizeExpressionText(Body body, string expressionText) =>
		CanonicalizeTextLiteralEscapes(
				NormalizeListImplementationNamesToPluralAliases(body, expressionText)).
			Replace(Type.ValueLowercase + ".", string.Empty, StringComparison.Ordinal).
			Replace(body.Method.Type.Name + ".", string.Empty, StringComparison.Ordinal);

	private static string CanonicalizeTextLiteralEscapes(string expressionText)
	{
		var builder = new StringBuilder(expressionText.Length);
		var insideText = false;
		for (var index = 0; index < expressionText.Length; index++)
		{
			var character = expressionText[index];
			if (character == '"')
			{
				insideText = !insideText;
				builder.Append(character);
				continue;
			}
			if (!insideText)
			{
				builder.Append(character);
				continue;
			}
			if (character == '\\')
			{
				if (index + 1 < expressionText.Length)
				{
					var nextCharacter = expressionText[index + 1];
					if (nextCharacter is 'n' or 'r' or 't' or '\\' or '"')
					{
						builder.Append('\\');
						builder.Append(nextCharacter);
						index++;
						continue;
					}
				}
				builder.Append(@"\\");
				continue;
			}
			builder.Append(character switch
			{
				'\n' => "\\n",
				'\r' => "\\r",
				'\t' => "\\t",
				_ => character.ToString()
			});
		}
		return builder.ToString();
	}

	private static string NormalizeListImplementationNamesToPluralAliases(Body body,
		string expressionText)
	{
		var normalizedText = expressionText;
		foreach (var typeEntry in body.Method.Type.Package.Types)
		{
			normalizedText = normalizedText.Replace($"{Type.List}({typeEntry.Key})", typeEntry.Key + "s",
				StringComparison.Ordinal);
			normalizedText = normalizedText.Replace(typeEntry.Key + "s.",
				string.Empty, StringComparison.Ordinal);
		}
		return normalizedText;
	}

	private sealed class GeneratedBinaryExpressionDoesNotMatchInputExactly(Body body,
		Expression binary,
		string inputText)
		: ParsingFailed(body, binary + ", inputText=" + inputText); //ncrunch: no coverage
}
