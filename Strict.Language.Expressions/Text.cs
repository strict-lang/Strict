namespace Strict.Language.Expressions
{
	public class Text : Value
	{
		public Text(Context context, string value) : base(context.GetType(Base.Text), value) { }
		public override string ToString() => "\"" + Data + "\"";
		
		public override bool Equals(Expression? other) =>
			other is Value v && (string)Data == (string)v.Data;
	}
}