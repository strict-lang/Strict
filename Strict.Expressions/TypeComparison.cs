using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Represents a right-hand side of an 'is'/'is not' comparison when the RHS is a type name.
/// This allows parsing 'customer is Customer' as a type check instead of trying to instantiate.
/// </summary>
public sealed class TypeComparison(Type returnType, Type targetType)
	: ConcreteExpression(returnType)
{
	public Type TargetType { get; } = targetType;
	public override bool IsConstant => true;
	public override string ToString() => TargetType.Name;

	/// <summary>
	/// If the given input represents a known type, returns a TypeComparison expression carrying
	/// that type (with ReturnType = Base.Type). Otherwise, falls back to normal expression parsing.
	/// </summary>
	public static Expression Parse(Body body, ReadOnlySpan<char> input, Range nextTokenRange)
	{
		var text = input[nextTokenRange].ToString();
		var foundType = body.ReturnType.FindType(text);
		if (foundType != null)
			return new TypeComparison(body.Method.GetType(Base.Type), foundType);
		return body.Method.ParseExpression(body, input[nextTokenRange]);
	}
}