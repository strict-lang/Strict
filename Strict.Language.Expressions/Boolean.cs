namespace Strict.Language.Expressions
{
    public class Boolean : Value
    {
        public Boolean(Context context, bool value) : base(context.GetType(Base.Boolean),
            value) { }

        public override string ToString() => base.ToString().ToLower();

        public override bool Equals(Expression? other) =>
            other is Value v && (bool)Data == (bool)v.Data;

        public static Expression? TryParse(Method context, string input) =>
            input switch
            {
                "true" => new Boolean(context, true),
                "false" => new Boolean(context, false),
                _ => null
            };
    }
}