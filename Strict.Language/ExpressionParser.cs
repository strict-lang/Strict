namespace Strict.Language;

/// <summary>
/// Abstracts the actual expressions and parsing away to the Expressions project.
/// <see cref="Method.GetBodyAndParseIfNeeded()"/> will call this lazily when it is called the
/// first time, which is not happening until a method is actually used.
/// This improves performance a lot as we almost do no parsing on any code (99%+ is not executed).
/// </summary>
public abstract class ExpressionParser
{
	public abstract Expression ParseLineExpression(Body body, ReadOnlySpan<char> line);
	public abstract Expression ParseExpression(Body body, ReadOnlySpan<char> text, bool makeMutable = false);
	public abstract List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> text);
	public abstract bool IsVariableMutated(Body body, string variableName);
}