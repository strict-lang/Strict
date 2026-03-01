using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Any expression with a fixed value, often optimized from all known code trees. Mostly used as
/// parameters and assignment values via the derived classes <see cref="Number"/>,
/// <see cref="Boolean"/> or <see cref="Text"/>. Already optimized for use in HighLevelRuntime.
/// </summary>
public class Value(Type valueType, ValueInstance data, int lineNumber = 0, bool isMutable = false)
	: ConcreteExpression(valueType, lineNumber, isMutable)
{
	protected Value(Type valueType, bool value, int lineNumber = 0, bool isMutable = false)
		: this(valueType, new ValueInstance(valueType, value), lineNumber, isMutable) { }

	protected Value(Type valueType, double value, int lineNumber = 0, bool isMutable = false)
		: this(valueType, new ValueInstance(valueType, value), lineNumber, isMutable) { }

	public Value(Type valueType, string text, int lineNumber = 0, bool isMutable = false)
		: this(valueType, new ValueInstance(text), lineNumber, isMutable) { }

	protected Value(Type valueType, List<ValueInstance> items, int lineNumber = 0,
		bool isMutable = false) : this(valueType, new ValueInstance(valueType, items), lineNumber,
		isMutable) { }

	public ValueInstance Data { get; } = data;
	public override string ToString() => Data.ToExpressionCodeString();
	public override bool IsConstant => true;
	public override bool Equals(Expression? other) => other is Value v && Data.Equals(v.Data);
	public override int GetHashCode() => HashCode.Combine(base.GetHashCode(), Data.GetHashCode());
}