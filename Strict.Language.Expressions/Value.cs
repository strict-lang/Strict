using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Any expression with a fixed value, often optimized from all known code trees. Mostly used as
/// parameters and assignment values via the derived classes <see cref="Number"/>,
/// <see cref="Boolean"/> or <see cref="Text"/>.
/// All expressions have a ReturnType and many expressions contains a <see cref="Value"/> like
/// <see cref="ConstantDeclaration"/> or indirectly as parts of a <see cref="Binary"/> expression.
/// For generic values like <see cref="List"/> the Data contains the generic type used.
/// </summary>
public class Value : ConcreteExpression
{
	public Value(Type valueType, object data) : base(valueType) => Data = data;
	public object Data { get; }
	public override string ToString() => Data.ToString()!;
	public override bool Equals(object? other) => Data.Equals(other);
	public override int GetHashCode() => HashCode.Combine(base.GetHashCode(), Data);
}