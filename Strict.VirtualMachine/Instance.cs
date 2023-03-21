using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

//TODO: Change this class so Value is always expression not an object! (LM)
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

	public static Instance operator +(Instance left, Instance right)
	{
		var leftReturnTypeName = left.TypeName;
		var rightReturnTypeName = right.TypeName;
		if (leftReturnTypeName == Base.Number && rightReturnTypeName == Base.Number)
			return new Instance(right.ReturnType ?? left.ReturnType,
				Convert.ToDouble(left.Value) + Convert.ToDouble(right.Value));
		if (leftReturnTypeName == Base.Text && rightReturnTypeName == Base.Text)
			return new Instance(right.ReturnType ?? left.ReturnType,
				left.Value.ToString() + right.Value);
		if (rightReturnTypeName == Base.Text && leftReturnTypeName == Base.Number)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		return !leftReturnTypeName.EndsWith('s')
			? new Instance(left.ReturnType, left.Value + right.Value.ToString())
			: AddElementToTheListAndGetInstance(left, right);
	}

	public static Instance operator -(Instance left, Instance right)
	{
		if (!left.TypeName.EndsWith('s'))
			return new Instance(left.ReturnType,
				Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value));
		var elements = new List<Expression>((List<Expression>)left.Value);
		if (right.Value is Expression rightExpression)
			elements.Remove(rightExpression);
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
		//TODO: Cover the tests for this case or delete (LM) ncrunch: no coverage start
		if (right.Value is Expression rightExpression)
		{
			elements.Add(rightExpression);
		}
		//ncrunch: no coverage end
		else
		{
			var rightValue = new Value(elements.First().ReturnType, right.Value);
			elements.Add(rightValue);
		}
		return new Instance(left.ReturnType, elements);
	}

	public override string ToString() => $"{Value} {TypeName}";
}