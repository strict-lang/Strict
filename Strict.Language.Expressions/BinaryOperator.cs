using System.Linq;

namespace Strict.Language.Expressions
{
	public static class BinaryOperator
	{
		public const char Plus = '+';
		public const char Minus = '-';
		public const char Multiply = '*';
		public const char Divide = '/';
		public const char Modulate = '%';
		public const char Assign = '=';
		public const char Smaller = '<';
		public const char Greater = '>';
		public static bool IsOperator(this char name) => All.Contains(name);

		private static readonly char[] All =
		{
			Plus, Minus, Multiply, Divide, Modulate, Assign, Smaller, Greater
		};
	}
	/*do with Pidgin
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

		public static bool IsBinaryOperator(this string name) =>
			name == Plus || name == Minus || name == Multiply || name == Divide || name == Modulate;

		private static readonly string[] All =
		{
			Plus, Minus, Multiply, Divide, Modulate, Open, Close, Assign, Smaller, Greater
		};
	}
	*/
}