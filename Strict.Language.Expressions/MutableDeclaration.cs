namespace Strict.Language.Expressions;

public sealed class MutableDeclaration(Body scope, string name, Expression value)
	: ConstantDeclaration(scope, name, value, true)
{
	public new static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal)
			? TryParseDeclaration(body, line, MutableWithSpaceAtEnd)
			: null;

	internal const string MutableWithSpaceAtEnd = Keyword.Mutable + " ";

	internal static Expression ParseMutableDeclarationWithValue(Body body, string name,
		ReadOnlySpan<char> valueSpan)
	{
		var value = valueSpan.IsFirstLetterUppercase() && (valueSpan.IsPlural() ||
			valueSpan.StartsWith(Base.List + '(' + Base.Mutable, StringComparison.Ordinal))
			? new List(body.Method.Type.GetType(valueSpan.ToString()))
			: body.Method.ParseExpression(body, valueSpan, true);
		return new MutableDeclaration(body, name, value);
	}

	public override string ToString() => MutableWithSpaceAtEnd + Name + " = " + Value;
}