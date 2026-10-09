namespace Strict.Language;

/// <summary>
/// Used in Body.Variables, each variable remembers the scope it was created it.
/// </summary>
public sealed class Variable(string name,
	bool isMutable,
	Expression initialValue,
	Body createdInScope,
	bool isImplicit = false)
{
	public string Name { get; } = name;
	public bool IsMutable { get; } = isMutable;
	public Expression InitialValue { get; internal set; } = initialValue;
	public Body CreatedInScope { get; } = createdInScope;
	public bool IsImplicit { get; } = isImplicit;
	public Type Type => InitialValue.ReturnType;

	public void CheckIfWeCouldUpdateValue(Expression value)
	{
		if (!IsMutable)
			throw new Body.ValueIsNotMutableAndCannotBeChanged(CreatedInScope, Name);
		if (!InitialValue.ReturnType.IsSameOrCanBeUsedAs(value.ReturnType))
			throw new NewExpressionDoesNotMatchVariableType(CreatedInScope, value, this);
	}

	public class
		NewExpressionDoesNotMatchVariableType(Body body, Expression newExpression, Variable variable)
		: ParsingFailed(body, newExpression.ToStringWithType() + " cannot be assigned to " + variable,
			variable.InitialValue.ReturnType);

	public override string ToString() => Name + " " + InitialValue.ReturnType;
}