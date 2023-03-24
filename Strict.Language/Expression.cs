namespace Strict.Language;

/// <summary>
/// Each line in a method is an expression, many expressions have child expressions (if, for,
/// while) or consists of multiple expressions (e.g. binary operations have two expressions).
/// There are no statements in Strict, every line in a method is an expression, every other
/// line in a .strict file is either implement, has or a method definition.
/// </summary>
public abstract class Expression : IEquatable<Expression>
{
	protected Expression(Type returnType, bool isMutable = false)
	{
		ReturnType = returnType;
		IsMutable = isMutable;
	}

	public Type ReturnType { get; }
	public bool IsMutable { get; set; }
	/// <summary>
	/// By default all expressions should be immutable in Strict. However, many times some part of the
	/// code will actually change something, thus making that expression AND anything that calls it
	/// mutable. Think of it as a virus that spreads all the way up. However if a high level
	/// expression is actually still immutable, it means everything it calls is also immutable and
	/// thus it can be evaluated once and will never change its value, a very important optimization.
	/// </summary>
	public bool ContainsAnythingMutable { get; protected set; } //ncrunch: no coverage

	public virtual bool Equals(Expression? other) =>
		!ReferenceEquals(null, other) &&
		(ReferenceEquals(this, other) || other.ToString() == ToString());

	public override bool Equals(object? obj) =>
		!ReferenceEquals(null, obj) && (ReferenceEquals(this, obj) ||
			obj.GetType() == GetType() && Equals((Expression)obj));

	public override int GetHashCode() => ToString().GetHashCode();
	public override string ToString() => base.ToString() + " " + ReturnType;

	public string ToStringWithType()
	{
		var text = ToString();
		return !text.EndsWith(ReturnType.ToString(), StringComparison.Ordinal)
			? text + " " + ReturnType
			: text;
	}
}