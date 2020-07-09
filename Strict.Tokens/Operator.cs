using System.Linq;

namespace Strict.Tokens
{
	public static class Operator
	{
		public const string Plus = "+";
		public const string Minus = "-";
		public const string Multiply = "*";
		public const string Divide = "/";
		public const string Modulate = "%";
		public const string Open = "(";
		public const string Close = ")";
		public const string Assign = "=";
		public const string Smaller = "<";
		public const string Greater = ">";
		public static bool IsOperator(this string name) => All.Contains(name);

		private static readonly string[] All =
		{
			Plus, Minus, Multiply, Divide, Modulate, Open, Close, Assign, Smaller, Greater
		};
	}
}