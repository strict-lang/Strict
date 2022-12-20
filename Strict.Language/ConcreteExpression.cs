namespace Strict.Language;

/// <summary>
/// Checks if the type used in an Assignment, MethodCall, MemberCall, ListCall, etc. is actually valid
/// and can be used directly. Generic types are not allowed as we can't operate on them directly.
/// </summary>
public abstract class ConcreteExpression : Expression
{
	protected ConcreteExpression(Type returnType, bool isMutable = false) : base(returnType, isMutable)
	{
		if (returnType.IsGeneric)
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(returnType,
				GetType().Name);
	}
}