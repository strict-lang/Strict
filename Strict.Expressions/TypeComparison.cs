using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Represents a right-hand side of an 'is'/'is not' comparison when the RHS is a type name.
/// This allows parsing 'customer is Customer' as a type check instead of trying to instantiate.
/// </summary>
public sealed class TypeComparison(Type returnType, Type targetType, int lineNumber = 0)
	: ConcreteExpression(returnType, lineNumber)
{
	public Type TargetType { get; } = targetType;
	public override bool IsConstant => true;
	public override string ToString() => TargetType.Name;

	/// <summary>
	/// If the given input represents a known type, returns a TypeComparison expression carrying
	/// that type (with ReturnType = Base.Type). Otherwise, fall back to normal expression parsing.
	/// </summary>
	public static Expression Parse(Body body, ReadOnlySpan<char> input, Range nextTokenRange)
	{
		var foundType = char.IsUpper(input[nextTokenRange.Start]) &&
			input[nextTokenRange.End.Value - 1].IsLetter()
				? body.ReturnType.FindType(input[nextTokenRange].ToString())
				: null;
		return foundType != null
			? new TypeComparison(body.Method.GetType(Base.Type), foundType, body.Method.TypeLineNumber + body.ParsingLineNumber)
			: body.Method.ParseExpression(body, input[nextTokenRange]);
	}
}