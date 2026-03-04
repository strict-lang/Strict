using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

/// <summary>
/// Runtime instance holding a type and its value. Value is polymorphic (number, text, bool,
/// List&lt;Expression&gt;, Dictionary&lt;Value,Value&gt;) until fully migrated to ValueInstance.
/// </summary>
public sealed class Instance
{
	public Instance(Type type, object value)
	{
		ReturnType = type!;
		Value = value is Value valueObj ? valueObj.Data : value;
	}

	public Instance(Expression expression)
	{
		ReturnType = expression.ReturnType!;
		Value = expression is Value value ? (object)value.Data : expression;
	}

	public Type ReturnType { get; }
	//TODO: migrate Value from object to ValueInstance once List<Expression>/Dictionary<Value,Value> usages are replaced
	private object rawValue = null!;
	public object Value
	{
		get => rawValue is ValueInstance vi
			? vi.IsText ? (object)vi.Text
			: vi.IsList || vi.IsDictionary ? rawValue // pass-through for complex types
			: ReturnType.IsBoolean ? vi.Number != 0
			: (object)vi.Number
			: rawValue;
		set => rawValue = value;
	}

	internal object ValueInstance => rawValue;

	/// <summary>Returns the backing ValueInstance if stored as one, otherwise default.</summary>
	internal Strict.Expressions.ValueInstance RawValueInstance =>
		rawValue is Strict.Expressions.ValueInstance vi ? vi : default;

	public object GetRawValue() => Value; // Value already unwraps ValueInstance

	private double ToDouble() =>
		rawValue is ValueInstance vi
			? vi.Number
			: Convert.ToDouble(rawValue);

	public static Instance operator +(Instance left, Instance right)
	{
		if (!left.ReturnType.IsList)
			return HandleTextTypeConversionForBinaryOperations(left, right, BinaryOperator.Plus);
		if (left.ReturnType is GenericTypeImplementation { Name: Type.List })
			return new Instance(left.ReturnType, left.Value.ToString() + right.Value);
		return AddElementToTheListAndGetInstance(left, right);
	}

	private static Instance HandleTextTypeConversionForBinaryOperations(Instance left,
		Instance right, string binaryOperator) =>
		left.ReturnType.IsNumber && right.ReturnType.IsNumber
			? new Instance(right.ReturnType,
				binaryOperator == BinaryOperator.Plus
					? left.ToDouble() + right.ToDouble()
					: left.ToDouble() - right.ToDouble())
			: left.ReturnType.IsText && right.ReturnType.IsText
				? new Instance(right.ReturnType, left.GetRawValue().ToString() + right.GetRawValue())
				: right.ReturnType.IsText && left.ReturnType.IsNumber
					? new Instance(right.ReturnType, left.GetRawValue().ToString() + right.GetRawValue())
					: new Instance(left.ReturnType, left.GetRawValue().ToString() + right.GetRawValue());

	public static Instance operator -(Instance left, Instance right)
	{
		if (!left.ReturnType.IsList)
			return new Instance(left.ReturnType, left.ToDouble() - right.ToDouble());
		var elements = new List<Expression>((List<Expression>)left.Value);
		if (right.rawValue is Expression rightExpression)
			elements.Remove(rightExpression);
		else
			elements.RemoveAt(elements.FindIndex(element =>
				BytecodeInterpreter.AreEqual(((Value)element).Data.IsText
					? ((Value)element).Data.Text
					: (object)((Value)element).Data.Number, right.Value)));
		return new Instance(left.ReturnType, elements);
	}

	public static bool operator >(Instance left, Instance right) =>
		left.ToDouble() > right.ToDouble();

	public static bool operator <(Instance left, Instance right) =>
		left.ToDouble() < right.ToDouble();

	private static Instance AddElementToTheListAndGetInstance(Instance left, Instance right)
	{
		var elements = new List<Expression>((List<Expression>)left.Value);
		var elementType = elements.First().ReturnType;
		var rightValue = right.rawValue is ValueInstance vi
			? new Value(elementType, vi)
			: elementType.IsNumber
				? new Value(elementType, new ValueInstance(elementType, Convert.ToDouble(right.Value)))
				: new Value(elementType, new ValueInstance(right.Value.ToString() ?? ""));
		elements.Add(rightValue);
		return new Instance(left.ReturnType, elements);
	}

	public override string ToString()
	{
		if (rawValue is ValueInstance vi && vi.IsList)
			return string.Join(" ", vi.List.Items.Select(item =>
				item.IsText ? item.Text
				: item.Number == Math.Truncate(item.Number)
					? ((long)item.Number).ToString()
					: item.Number.ToString()));
		var v = Value;
		if (v is List<Expression> list)
			return string.Join(" ", list.Select(e => e is Value val
				? (val.Data.IsText ? val.Data.Text : val.Data.Number == Math.Truncate(val.Data.Number)
					? ((long)val.Data.Number).ToString()
					: val.Data.Number.ToString())
				: e.ToString()));
		if (v is double d)
			return $"{(d == Math.Truncate(d) ? (long)d : d)}";
		return v?.ToString() ?? "";
	}
}