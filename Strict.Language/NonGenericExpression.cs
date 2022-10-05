namespace Strict.Language;

/// <summary>
/// Checks if the type used in an Assignment, MethodCall, MemberCall, ListCall, etc. is actually valid
/// and can be used directly. Generic types are not allowed as we can't operate on them directly.
/// </summary>
public abstract class NonGenericExpression : Expression
{
	protected NonGenericExpression(Type returnType) : base(returnType)
	{
		if (returnType.IsGeneric)
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(returnType,
				GetType().Name);
	}
}