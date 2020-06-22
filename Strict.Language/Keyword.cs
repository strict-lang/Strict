using System.Linq;

namespace Strict.Language
{
	public static class Keyword
	{
		public const string Has = "has";
		public const string As = "as";
		public const string Assert = "assert";
		public const string Test = "test";
		public const string Create = "create";
		public const string Throw = "throw";
		public const string Method = "method";
		public const string Returns = "returns";
		public const string Is = "is";
		public const string Isnt = "isnt";
		public const string Not = "not";
		public const string True = "true";
		public const string False = "false";
		public const string Continue = "continue";
		public const string Break = "break";
		public const string Do = "do";
		public const string In = "in";
		public const string From = "from";
		public const string Return = "return";
		public const string To = "to";
		public const string For = "for";
		public const string If = "if";
		public const string Else = "else";
		public const string Yield = "yield";

		public static bool IsKeyword(string name) => All.Contains(name);

		private static readonly string[] All =
		{
			Has, As, Assert, Test, Create, Throw, Method, Returns, Is, Isnt, Not, True, False,
			Continue, Break, Do, In, From, Return, To, For, If, Else, Yield
		};
	}
}