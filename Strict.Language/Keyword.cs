namespace Strict.Language;

public static class Keyword
{
	public const string Has = "has";
	public const string Mutable = "mutable";
	public const string Constant = "constant";
	public const string If = "if";
	public const string Else = "else";
	public const string For = "for";
	public const string With = "with";
	public const string True = "true";
	public const string False = "false";
	public const string Return = "return";
	public static readonly string[] GetAllKeywords =
	{
		Has, Mutable, Constant, If, Else, For, With, True, False, Return
	};
}