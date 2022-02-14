namespace Strict.Language.Expressions;

public class Boolean : Value
{
	public Boolean(Context context, bool value) : base(context.GetType(Base.Boolean), value) { }
	public override string ToString() => base.ToString().ToLower();

	public override bool Equals(Expression? other) =>
		other is Value v && (bool)Data == (bool)v.Data;

	public static Expression? TryParse(Method.Line line, string partToParse) =>
		partToParse switch
		{
			"true" => new Boolean(line.Method, true),
			"false" => new Boolean(line.Method, false),
			_ => null
		};
}