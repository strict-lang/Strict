using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

/// <summary>
///   The only place where we can have a "static" method call to one of the from methods of a type
///   before we have a type instance yet, it is the only way to create instances.
/// </summary>
public sealed class Instance
{
	private readonly string typeName = string.Empty;

	public Instance(Type? type, object value)
	{
		ReturnType = type;
		if (value is Value valueObj)
			Value = valueObj.Data;
		else
			Value = value;
	}

	public Instance(string typeName, object value)
	{
		Value = value;
		this.typeName = typeName;
	}

	public Instance(Expression expression, bool isMember = false)
	{
		ReturnType = expression.ReturnType;
		if (expression is Value value)
			Value = value.Data;
		else
			Value = expression;
		IsMember = isMember;
	}

	public bool IsMember { get; }
	public Type? ReturnType { get; }
	public string TypeName =>
		ReturnType == null
			? typeName
			: ReturnType.Name;
	public object Value { get; set; }

	public object GetRawValue()
	{
		if (Value is Value value)
			return value.Data;
		return Value;
	}

	public static Instance operator +(Instance left, Instance right)
	{
		if (!left.TypeName.StartsWith(Base.List, StringComparison.Ordinal))
			return HandleTextTypeConversionForBinaryOperations(left, right, BinaryOperator.Plus);
		return left.ReturnType is GenericTypeImplementation { Name: Base.List }
			? new Instance(left.ReturnType, left.Value + right.Value.ToString())
			: AddElementToTheListAndGetInstance(left, right);
	}

	private static Instance HandleTextTypeConversionForBinaryOperations(Instance left,
		Instance right, string binaryOperator)
	{
		var leftReturnTypeName = left.TypeName;
		var rightReturnTypeName = right.TypeName;
		if (leftReturnTypeName == Base.Number && rightReturnTypeName == Base.Number)
			return new Instance(right.ReturnType ?? left.ReturnType,
				binaryOperator == BinaryOperator.Plus
					? Convert.ToDouble(left.Value) + Convert.ToDouble(right.Value)
					: Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value));
		if (leftReturnTypeName == Base.Text && rightReturnTypeName == Base.Text)
			return new Instance(right.ReturnType ?? left.ReturnType,
				left.Value.ToString() + right.Value);
		if (rightReturnTypeName == Base.Text && leftReturnTypeName == Base.Number)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		return new Instance(left.ReturnType, left.Value + right.Value.ToString());
	}

	public static Instance operator -(Instance left, Instance right)
	{
		if (!left.TypeName.StartsWith("List", StringComparison.Ordinal))
			return new Instance(left.ReturnType,
				Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value));
		var elements = new List<Expression>((List<Expression>)left.Value);
		if (right.Value is Expression rightExpression)
		{
			elements.Remove(rightExpression);
		}
		else
		{
			var indexToRemove =
				elements.FindIndex(element => ((Value)element).Data.Equals(right.Value));
			elements.RemoveAt(indexToRemove);
		}
		return new Instance(left.ReturnType, elements);
	}

	public static bool operator >(Instance left, Instance right) =>
		Convert.ToDouble(left.Value) > Convert.ToDouble(right.Value);

	public static bool operator <(Instance left, Instance right) =>
		Convert.ToDouble(left.Value) < Convert.ToDouble(right.Value);

	private static Instance AddElementToTheListAndGetInstance(Instance left, Instance right)
	{
		var elements = new List<Expression>((List<Expression>)left.Value);
		var rightValue = new Value(elements.First().ReturnType, right.Value);
		elements.Add(rightValue);
		return new Instance(left.ReturnType, elements);
	}

	public override string ToString() => $"{Value} {TypeName}";
}