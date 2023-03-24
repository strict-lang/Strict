namespace Strict.Language.Expressions;

public sealed class MutableDeclaration : ConstantDeclaration
{
	public MutableDeclaration(Body scope, string name, Expression value) :
		base(scope, name, value, true) { }

	internal static Expression ParseMutableDeclarationWithValue(Body body, string name, ReadOnlySpan<char> valueSpan)
	{
		var value = valueSpan.IsFirstLetterUppercase() && (valueSpan.IsPlural() || valueSpan.StartsWith(Base.List + '(' + Base.Mutable, StringComparison.Ordinal))
			? new List(body.Method.Type.GetType(valueSpan.ToString()))
			: body.Method.ParseExpression(body, valueSpan);
		value.IsMutable = true;
		return new MutableDeclaration(body, name, value);
	}

	public override string ToString() =>
		MutableWithSpaceAtEnd + Name + " = " + Value;

	internal const string MutableWithSpaceAtEnd = Keyword.Mutable + " ";
}