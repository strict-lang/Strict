using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Represents a type name used as a pattern in a type-dispatch selector-if expression. E.g. in
/// "if generic is Number then ...", the variable is this Number TypePattern in the then-branch.
/// </summary>
internal sealed class TypePattern(Type concreteType, string displayName, int lineNumber = 0)
	: Expression(concreteType, lineNumber)
{
	public override string ToString() => displayName;
	//ncrunch: no coverage start
	public override bool IsConstant => true;

	public override bool Equals(Expression? other) =>
		other is TypePattern tp && ReturnType == tp.ReturnType;

	public override int GetHashCode() => ReturnType.GetHashCode();
}