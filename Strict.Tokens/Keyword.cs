using System.Linq;

namespace Strict.Tokens
{
	public static class Keyword
	{
		public const string Implement = "implement";
		public const string Has = "has";
		public const string Let = "let";
		public const string As = "as";
		/// <summary>
		/// Points to the class instance we are currently in
		/// </summary>
		public const string Self = "self";
		public const string Test = "test";
		public const string Throw = "throw";
		public const string Is = "is";
		public const string Not = "not";
		public const string True = "true";
		public const string False = "false";
		public const string Break = "break";
		public const string In = "in";
		public const string From = "from";
		public const string To = "to";
		public const string For = "for";
		public const string If = "if";
		public const string Else = "else";
		public const string While = "while";
		public const string Return = "return";
		public const string Yield = "yield";
		public const string Returns = "returns";
		public static bool IsKeyword(this string name) => All.Contains(name);

		public static bool IsKeywordFunction(this string name) =>
			name == From || name == Is || name == To;

		private static readonly string[] All =
		{
			Has, Let, As, Test, Throw, Is, Not, True, False, Break, While, In, From, Return, To,
			For, If, Else, Yield, Returns
		};
	}
}