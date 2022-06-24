namespace Strict.Language.Expressions;

public sealed class List : Value
{
	public List(Context context, string value) : base(context.GetType(Base.List), value) { }
	public override string ToString() => "(" + Data + ")";

	public static Expression? TryParse(Method.Line line, string input) =>
		input.Length > 2 && input[0] == '(' && input[^1] == ')'
			? new List(line.Method, input[1..^1]) // TODO: input should be converted to list of types instead of storing it as string
			: null;
}