using System;

namespace Strict.Language.Expressions;

public class MutableDeclaration : ConstantDeclaration
{
	public MutableDeclaration(Body scope, string name, Expression value) :
		base(scope, name, value, true) { }

	internal static Expression ParseMutableDeclarationWithValue(Body body, string name, ReadOnlySpan<char> valueSpan)
	{
		var value = valueSpan.IsFirstLetterUppercase() && valueSpan.IsPlural()
			? new List(body.Method.Type.GetType(valueSpan.ToString()))
			: body.Method.ParseExpression(body, valueSpan);
		value.IsMutable = true;
		return new MutableDeclaration(body, name, value);
	}

	public override string ToString() =>
		Mutable + Name + " = " + Value;

	internal const string Mutable = "mutable ";
}