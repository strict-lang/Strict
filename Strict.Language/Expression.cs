using System;

namespace Strict.Language
{
	/// <summary>
	/// Each line in a method is an expression, many expressions have child expressions (if, for,
	/// while) or consists of multiple expressions (e.g. binary operations have two expressions).
	/// There are no statements in Strict, every line in a method is an expression, every other
	/// line in a .strict file is either implement, has or a method definition.
	/// </summary>
	public abstract class Expression : IEquatable<Expression>
	{
		protected Expression(Type returnType) => ReturnType = returnType;
		public Type ReturnType { get; }
		
		public virtual bool Equals(Expression? other) =>
			!ReferenceEquals(null, other) &&
			(ReferenceEquals(this, other) || other.ToString() == ToString());

		public override bool Equals(object? obj) =>
			!ReferenceEquals(null, obj) && (ReferenceEquals(this, obj) ||
				obj.GetType() == GetType() && Equals((Expression)obj));

		public override int GetHashCode() => ToString()!.GetHashCode();

		public override string ToString() => base.ToString() + " " + ReturnType;
	}
}