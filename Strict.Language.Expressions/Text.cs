namespace Strict.Language.Expressions
{
    public class Text : Value
    {
        public Text(Context context, string value) :
            base(context.GetType(Base.Text), value) { }

        public override string ToString() => "\"" + Data + "\"";

        public override bool Equals(Expression? other) =>
            other is Value v && (string)Data == (string)v.Data;

        public static Expression? TryParse(Method context, string input) =>
            input.Length >= 2 && input[0] == '"' && input[^1] == '"'
                ? new Text(context, input[1..^1])
                : null;
    }
}