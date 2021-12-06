using System.Linq;

namespace Strict.Language.Expressions;

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