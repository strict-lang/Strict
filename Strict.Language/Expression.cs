using System;

namespace Strict.Language;

/// <summary>
/// Each line in a method is an expression, many expressions have child expressions (if, for,
/// while) or consists of multiple expressions (e.g., binary operations have two expressions).
/// There are no statements in Strict, every line in a method is an expression, every other
/// line in a .strict file is either implement, has, or a method definition.
/// </summary>
public abstract class Expression(Type returnType, int lineNumber = 0, bool isMutable = false)
	: IEquatable<Expression>
{
	public Type ReturnType { get; } = returnType;
	public int LineNumber { get; } = lineNumber;
	public bool IsMutable { get; } = isMutable;
	/// <summary>
	/// By default, all expressions should be immutable in Strict. However, many times some part of
	/// the code will actually change something, thus making that expression AND anything that calls
	/// it mutable. Think of it as a virus that spreads all the way up. However, if a high-level
	/// expression is actually still immutable, it means everything it calls is also immutable, and
	/// thus it can be evaluated once and will never change its value, a very important optimization.
	/// </summary>
	public bool ContainsAnythingMutable
	{
		get
		{
			if (IsMutable)
				return true;
			if (this is Body body)
				foreach (var bodyExpression in body.Expressions)
					if (bodyExpression.ContainsAnythingMutable)
						return true;
			return false;
		}
	}
	public abstract bool IsConstant { get; }

	public virtual bool Equals(Expression? other) =>
		!ReferenceEquals(null, other) &&
		(ReferenceEquals(this, other) || other.ToString() == ToString());

	public override bool Equals(object? obj) =>
		!ReferenceEquals(null, obj) && (ReferenceEquals(this, obj) ||
			obj.GetType() == GetType() && Equals((Expression)obj));

	public override int GetHashCode() => ToString().GetHashCode();
	public override string ToString() => base.ToString() + " " + ReturnType;

	protected static string IndentExpression(Expression expression) =>
		'\t' + expression.ToString().Replace(Environment.NewLine, Environment.NewLine + '\t');

	public string ToStringWithType()
	{
		var text = ToString();
		return !text.EndsWith(ReturnType.ToString(), StringComparison.Ordinal)
			? text + " " + ReturnType
			: text;
	}
}