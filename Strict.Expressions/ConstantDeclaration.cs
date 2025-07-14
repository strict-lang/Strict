using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// Constant declares a variable in a method, often a fixed value that is optimized away.
/// </summary>
public class ConstantDeclaration : ConcreteExpression
{
	public ConstantDeclaration(Body scope, string name, Expression value, bool isMutable = false) :
		base(value.ReturnType, isMutable)
	{
		if (!name.IsWord())
			throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		scope.AddVariable(name, value, isMutable);
		Name = name;
		Value = value;
	}

	public string Name { get; }
	public Expression Value { get; }
	public override int GetHashCode() => Name.GetHashCode() ^ Value.GetHashCode();
	public override string ToString() => ConstantWithSpaceAtEnd + Name + " = " + Value;
	internal const string ConstantWithSpaceAtEnd = Keyword.Constant + " ";

	public override bool Equals(Expression? other) =>
		other is ConstantDeclaration a && Equals(Name, a.Name) && Value.Equals(a.Value);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(ConstantWithSpaceAtEnd, StringComparison.Ordinal)
			? TryParseDeclaration(body, line, ConstantWithSpaceAtEnd)
			: null;

	/// <summary>
	/// Highly optimized parsing of assignments, skips over the mutable, grabs the name of the local
	/// variable, then skips over the space, equal and space characters and parses the rest, e.g.
	/// constant hello = "hello" + " " + "world"
	///					 ^ ^       ^ ^   ^ ^       END, using TryParseExpression with Range(12, 35)
	/// </summary>
	protected static Expression TryParseDeclaration(Body body, ReadOnlySpan<char> line,
		string declarationType)
	{
		var parts = line[declarationType.Length..].Split();
		parts.MoveNext();
		var name = parts.Current.ToString();
		if (!parts.MoveNext() || !parts.MoveNext())
			throw new MissingAssignmentValueExpression(body);
		var valueSpan = line[(declarationType.Length + name.Length + 1 + 1 + 1)..];
		return declarationType == ConstantWithSpaceAtEnd
			? ParseConstantDeclarationWithValue(body, name, valueSpan)
			: MutableDeclaration.ParseMutableDeclarationWithValue(body, name, valueSpan);
	}

	private static Expression ParseConstantDeclarationWithValue(Body body, string name,
		ReadOnlySpan<char> valueSpan) =>
		new ConstantDeclaration(body, name, body.Method.ParseExpression(body, valueSpan, false));

	public sealed class MissingAssignmentValueExpression(Body body) : ParsingFailed(body);
}