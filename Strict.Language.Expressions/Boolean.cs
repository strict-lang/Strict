namespace Strict.Language.Expressions
{
	public class Boolean : Value
	{
		public Boolean(Context context, bool value) :
			base(context.GetType(Base.Boolean), value) { }

		public override string ToString() => Data.ToString()!.ToLower();

		public override bool Equals(Expression? other) =>
			other is Value v && (bool)Data == (bool)v.Data;
	}
}