using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

/// <summary>
/// The only place where we can have a "static" method call to one of the from methods of a type
/// before we have a type instance yet, it is the only way to create instances.
/// </summary>
public sealed class Instance
{
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
		TypeName = typeName;
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

	//TODO: very inefficient, object should be avoided like we did in Value and ValueInstance, also we don't need that much stuff here, figure out more efficient solution!
	public bool IsMember { get; }
	public Type ReturnType { get; }
	public object Value { get; set; }

	public object GetRawValue()
	{
		if (Value is Value value)
			return value.Data;
		return Value;
	}

	public static Instance operator +(Instance left, Instance right)
	{
		if (!left.ReturnType.IsList)
			return HandleTextTypeConversionForBinaryOperations(left, right, BinaryOperator.Plus);
		//TODO: over complicated, should always be a list here
		if (left.ReturnType is GenericTypeImplementation { Name: Base.List })
		{
			return new Instance(left.ReturnType, left.Value + right.Value.ToString());
		}
		else
		{
			return AddElementToTheListAndGetInstance(left, right);
		}
	}

	private static Instance HandleTextTypeConversionForBinaryOperations(Instance left,
		Instance right, string binaryOperator)
	{
		if (left.ReturnType.IsNumber && right.ReturnType.IsNumber)
			return new Instance(right.ReturnType ?? left.ReturnType,
				binaryOperator == BinaryOperator.Plus
					? Convert.ToDouble(left.Value) + Convert.ToDouble(right.Value)
					: Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value));
		if (left.ReturnType.IsText && right.ReturnType.IsText)
			return new Instance(right.ReturnType ?? left.ReturnType,
				left.Value.ToString() + right.Value);
		if (right.ReturnType.IsText && left.ReturnType.IsNumber)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		return new Instance(left.ReturnType, left.Value + right.Value.ToString());
	}

	public static Instance operator -(Instance left, Instance right)
	{
		if (!left.ReturnType.IsList)
			return new Instance(left.ReturnType,
				Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value));
		var elements = new List<Expression>((List<Expression>)left.Value);
		if (right.Value is Expression rightExpression)
			elements.Remove(rightExpression);
		else
			elements.RemoveAt(elements.FindIndex(element => ((Value)element).Data.Equals(right.Value)));
		return new Instance(left.ReturnType, elements);
	}

	public static bool operator >(Instance left, Instance right)
	{
		return Convert.ToDouble(left.Value) > Convert.ToDouble(right.Value);
	}

	public static bool operator <(Instance left, Instance right)
	{
		return Convert.ToDouble(left.Value) < Convert.ToDouble(right.Value);
	}

	private static Instance AddElementToTheListAndGetInstance(Instance left, Instance right)
	{
		var elements = new List<Expression>((List<Expression>)left.Value);
		var rightValue = new Value(elements.First().ReturnType, right.Value);
		elements.Add(rightValue);
		return new Instance(left.ReturnType, elements);
	}

	public override string ToString() => $"{Value} {ReturnType.Name}";
}