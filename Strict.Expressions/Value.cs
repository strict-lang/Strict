using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Any expression with a fixed value, often optimized from all known code trees. Mostly used as
/// parameters and assignment values via the derived classes <see cref="Number"/>,
/// <see cref="Boolean"/> or <see cref="Text"/>.
/// All expressions have a ReturnType and many expressions contains a <see cref="Value"/> like
/// <see cref="Declaration"/> or indirectly as parts of a <see cref="Binary"/> expression.
/// For generic values like <see cref="List"/> the Data contains the generic type used.
/// </summary>
public class Value(Type valueType, object data, int lineNumber = 0, bool isMutable = false)
	: ConcreteExpression(valueType, lineNumber, isMutable)
{
	public object Data { get; } = data;

	public override string ToString() =>
		Data is string
			? "\"" + Data + "\""
			: Data is double doubleData
				? doubleData.ToString("0.0")
				: Data.ToString()!;

	public override bool IsConstant => true;

	public override bool Equals(Expression? other) =>
		other is Value v && EqualsExtensions.AreEqual(Data, v.Data);

	public override int GetHashCode() => HashCode.Combine(base.GetHashCode(), Data);
}