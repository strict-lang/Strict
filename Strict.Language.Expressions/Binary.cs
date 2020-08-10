namespace Strict.Language.Expressions
{
	public class Binary : MethodCall
	{
		public Binary(Expression left, Method operatorMethod, Expression right) : base(left,
			operatorMethod, right) { }

		public Expression Left => Instance;
		public Expression Right => Arguments[0];
		public override string ToString() => Left + " " + Method.Name + " " + Right;
	}
}