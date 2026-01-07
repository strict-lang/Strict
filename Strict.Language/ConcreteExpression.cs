namespace Strict.Language;

/// <summary>
/// Checks if the type used in an ConstantDeclaration, MethodCall, MemberCall, ListCall, etc. is
/// actually valid and can be used directly. Generic types are not allowed as we can't operate on
/// them directly.
/// </summary>
public abstract class ConcreteExpression : Expression
{
	protected ConcreteExpression(Type returnType, int lineNumber = 0, bool isMutable = false) : base(returnType, lineNumber, isMutable)
	{
		if (returnType.IsGeneric)
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(returnType, GetType().Name);
	}
}