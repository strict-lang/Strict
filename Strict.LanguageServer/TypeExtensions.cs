using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public static class TypeExtensions
{
	public static Expression ParseExpression(this Type type, params string?[] lines)
	{
		var methodLines = new string[lines.Length + 1];
		methodLines[0] = "Run";
		for (var index = 0; index < lines.Length; index++)
			methodLines[index + 1] = '\t' + lines[index];
		return new Method(type, 0, new MethodExpressionParser(), methodLines).
			GetBodyAndParseIfNeeded();
	}
}