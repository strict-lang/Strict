namespace Strict.Language.Expressions;

public sealed class Error(Expression message)
	: Expression(message.ReturnType.GetType(Base.Error));