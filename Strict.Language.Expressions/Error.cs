namespace Strict.Language.Expressions;

public sealed class Error : Expression
{
	public Error(Expression message) : base(message.ReturnType.GetType(Base.Error)) { }
}