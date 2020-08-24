using System;

namespace Strict.Language.Expressions
{
	/// <summary>
	/// Let statements in strict, which usually assigns a fixed value that is optimized away.
	/// </summary>
	public class Assignment : Expression
	{
		public Assignment(Identifier name, Expression value) : base(value.ReturnType)
		{
			Name = name;
			Value = value;
		}

		public Identifier Name { get; }
		public Expression Value { get; }

		public override string ToString() => "let " + Name + " = " + Value;

		public override bool Equals(Expression? other) =>
			other is Assignment a && Equals(Name, a.Name) && Value.Equals(a.Value);

		public static Expression? TryParse(Method context, string input) =>
			input.StartsWith("let ") && TryLetExpression(context, input, out var name, out var value)
				? new Assignment(name!, value!)
				: null;

		private static bool TryLetExpression(Method context, string input, out Identifier? name, out Expression? value)
		{
			name = null;
			value = null;
			string[] parts = input.Split(new[] { "let ", " = " }, StringSplitOptions.RemoveEmptyEntries);
			if (parts.Length != 2)
				return false;
			value = MethodExpressionParser.TryParse(context, parts[1]);
			if (value == null)
				throw new InvalidExpression(input);
			name = Identifier.TryParse(parts[0], value.ReturnType);
			return name != null;
		}

		private class InvalidExpression : Exception
		{
			public InvalidExpression(string input) : base(input) { }
		}
	}
}