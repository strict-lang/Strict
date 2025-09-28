using Strict.Language;

namespace Strict.Expressions;

public sealed class Error(Expression message)
	: Expression(message.ReturnType.GetType(Base.Error))
{
	public override bool IsConstant => message.IsConstant;
	public override string ToString() => "Error(" + message + ")";
}