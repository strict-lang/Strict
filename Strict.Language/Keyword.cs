namespace Strict.Language;

public static class Keyword
{
	/// <summary>
	/// "has" defines members at type level and for structure and constraints.
	/// </summary>
	public const string Has = "has";
	/// <summary>
	/// "constant" is for pure, constraint-free values, anything at type level or inside methods that
	/// pretty much can be optimized away as the value is known and not changing (e.g. enum values).
	/// </summary>
	public const string Constant = "constant";
	/// <summary>
	/// By far the rarest usage of members in a type or inside methods to have changable values,
	/// mostly important for optimizations and done implicitly like the mutable index for loops.
	/// </summary>
	public const string Mutable = "mutable";
	public const string If = "if";
	public const string Else = "else";
	public const string For = "for";
	public const string With = "with";
	public const string Return = "return";
	public static readonly string[] GetAllKeywords =
	[
		Has, Constant, Mutable, If, Else, For, With, Return
	];
}