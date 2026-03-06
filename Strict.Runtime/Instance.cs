using System.Globalization;
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
	/// <summary>
	/// Legacy constructor still used during migration to ValueInstance-based execution.
	/// </summary>
	public Instance(Type type, object value)
	{
		ReturnType = type;
		Value = value switch
		{
			List listExpression => listExpression.Values,
			Value valueObj => valueObj.Data,
			_ => value
		};
	}

	public Instance(Expression expression)
	{
		ReturnType = expression.ReturnType;
		Value = expression switch
		{
			List listExpression => listExpression.Values,
			Value value => value.Data,
			_ => expression
		};
	}

	public Type ReturnType { get; }
	private object rawValue = null!;
	private ValueInstance data;
	public object Value
	{
		get =>
			rawValue is ValueInstance vi
				? vi.IsText
					? vi.Text
					: vi.IsList
						? ToExpressionList(vi)
						: vi.IsDictionary
							? ToValueDictionary(vi)
							: ReturnType.IsBoolean
								? vi.Number != 0
								: vi.Number
				: rawValue;
		set
		{
			rawValue = value;
			data = ToValueInstance(value);
		}
	}
	internal ValueInstance ValueInstance => data;

	private ValueInstance ToValueInstance(object raw) =>
		raw is ValueInstance valueInstance
			? valueInstance
			: raw is Value value
				? value.Data
				: raw is MemberCall memberCall && memberCall.Member.InitialValue is Value initialValue
					? initialValue.Data
					: raw is IList<Expression> expressionList
						? new ValueInstance(ReturnType, expressionList.Select(expression =>
							expression is Value itemValue
								? itemValue.Data
								: expression is MemberCall call &&
								call.Member.InitialValue is Value innerInitialValue
									? innerInitialValue.Data
									: new ValueInstance(expression.ToString())).ToList())
						: raw is Dictionary<Value, Value> dictionary
							? new ValueInstance(ReturnType,
								dictionary.ToDictionary(keyValue => keyValue.Key.Data,
									keyValue => keyValue.Value.Data))
							: ReturnType.IsText
								? new ValueInstance(raw.ToString() ?? "")
								: new ValueInstance(ReturnType, Convert.ToDouble(raw));

	private List<Expression> ToExpressionList(ValueInstance listInstance)
	{
		var items = listInstance.List.Items;
		var listType = listInstance.List.ReturnType;
		var elementType = listType is GenericTypeImplementation
		{
			Generic.Name: Type.List
		} genericList
			? genericList.ImplementationTypes[0]
			: ReturnType;
		return items.Select(item => (Expression)new Value(item.IsText
			? elementType.GetType(Type.Text)
			: item.GetTypeExceptText(), item)).ToList();
	}

	private Dictionary<Value, Value> ToValueDictionary(ValueInstance dictionaryInstance)
	{
		var dictionary = new Dictionary<Value, Value>();
		foreach (var item in dictionaryInstance.GetDictionaryItems())
			dictionary[new Value(item.Key.IsText
				? ReturnType.GetType(Type.Text)
				: item.Key.GetTypeExceptText(), item.Key)] = new Value(item.Value.IsText
				? ReturnType.GetType(Type.Text)
				: item.Value.GetTypeExceptText(), item.Value);
		return dictionary;
	}

	private double ToDouble() =>
		data.IsText
			? rawValue is string
				? Convert.ToDouble(data.Text)
				: Convert.ToDouble(rawValue)
			: data.Number;

	public static Instance operator +(Instance left, Instance right)
	{
		if (left.data.IsList || left.Value is List<Expression>)
			return AddElementToTheListAndGetInstance(left, right);
		if (!left.ReturnType.IsList)
			return HandleTextTypeConversionForBinaryOperations(left, right, BinaryOperator.Plus);
		return left.ReturnType is GenericTypeImplementation { Name: Type.List }
			? new Instance(left.ReturnType, left.Value.ToString() + right.Value)
			: AddElementToTheListAndGetInstance(left, right);
	}

	private static Instance HandleTextTypeConversionForBinaryOperations(Instance left,
		Instance right, string binaryOperator) =>
		left.ReturnType.IsNumber && right.ReturnType.IsNumber
			? new Instance(right.ReturnType, binaryOperator == BinaryOperator.Plus
				? left.ToDouble() + right.ToDouble()
				: left.ToDouble() - right.ToDouble())
			: left.ReturnType.IsText && right.ReturnType.IsText
				? new Instance(right.ReturnType, left.GetRawValue().ToString() + right.GetRawValue())
				: right.ReturnType.IsText && left.ReturnType.IsNumber
					? new Instance(right.ReturnType, left.GetRawValue().ToString() + right.GetRawValue())
					: new Instance(left.ReturnType, left.GetRawValue().ToString() + right.GetRawValue());

	public static Instance operator -(Instance left, Instance right)
	{
		if (left.data.IsList)
		{
			var items = new List<ValueInstance>(left.data.List.Items);
			var removeIndex = items.FindIndex(item => BytecodeInterpreter.AreEqual(item.IsText
				? item.Text
				: item.Number, right.Value));
			if (removeIndex >= 0)
				items.RemoveAt(removeIndex);
			return new Instance(left.ReturnType, new ValueInstance(left.data.List.ReturnType, items));
		}
		if (left.Value is List<Expression> list)
		{
			var elementsCopy = new List<Expression>(list);
			if (right.rawValue is Expression rightExpression)
				elementsCopy.Remove(rightExpression);
			else
				elementsCopy.RemoveAt(elementsCopy.FindIndex(element => BytecodeInterpreter.AreEqual(
					((Value)element).Data.IsText
						? ((Value)element).Data.Text
						: ((Value)element).Data.Number, right.Value)));
			return new Instance(left.ReturnType, elementsCopy);
		}
		if (!left.ReturnType.IsList)
			return new Instance(left.ReturnType, left.ToDouble() - right.ToDouble());
		var listElements = new List<Expression>((List<Expression>)left.Value);
		if (right.rawValue is Expression expression)
			listElements.Remove(expression);
		else
			listElements.RemoveAt(listElements.FindIndex(element => BytecodeInterpreter.AreEqual(
				((Value)element).Data.IsText
					? ((Value)element).Data.Text
					: ((Value)element).Data.Number, right.Value)));
		return new Instance(left.ReturnType, listElements);
	}

	public static bool operator >(Instance left, Instance right)
	{
		return left.ToDouble() > right.ToDouble();
	}

	public static bool operator <(Instance left, Instance right)
	{
		return left.ToDouble() < right.ToDouble();
	}

	private static Instance AddElementToTheListAndGetInstance(Instance left, Instance right)
	{
		if (left.data.IsList)
		{
			var items = new List<ValueInstance>(left.data.List.Items);
			if (right.data.GetTypeExceptText().IsSameOrCanBeUsedAs(right.ReturnType))
				items.Add(right.data);
			else if (right.ReturnType.IsNumber)
				items.Add(new ValueInstance(right.ReturnType, Convert.ToDouble(right.Value)));
			else if (right.ReturnType.IsText)
				items.Add(new ValueInstance(right.Value.ToString() ?? ""));
			else
				items.Add(new ValueInstance(right.Value.ToString() ?? ""));
			return new Instance(left.ReturnType, new ValueInstance(left.data.List.ReturnType, items));
		}
		var elements = new List<Expression>((List<Expression>)left.Value);
		var elementType = elements.First().ReturnType;
		var rightValue = right.data.GetTypeExceptText().IsSameOrCanBeUsedAs(right.ReturnType)
			? new Value(elementType, right.data)
			: elementType.IsNumber
				? new Value(elementType, new ValueInstance(elementType, Convert.ToDouble(right.Value)))
				: new Value(elementType, new ValueInstance(right.Value.ToString() ?? ""));
		elements.Add(rightValue);
		return new Instance(left.ReturnType, elements);
	}

	public object GetRawValue() => Value;

	public override string ToString()
	{
		if (data.IsList)
			return string.Join(" ", data.List.Items.Select(item => item.IsText
				? item.Text
				: item.Number == Math.Truncate(item.Number)
					? ((long)item.Number).ToString()
					: item.Number.ToString(CultureInfo.InvariantCulture)));
		var v = Value;
		if (v is List<Expression> list)
			return string.Join(" ", list.Select(e => e is Value val
				? val.Data.IsText
					? val.Data.Text
					: val.Data.Number == Math.Truncate(val.Data.Number)
						? ((long)val.Data.Number).ToString()
						: val.Data.Number.ToString(CultureInfo.InvariantCulture)
				: e.ToString()));
		if (v is double d)
			return $"{
				(d == Math.Truncate(d)
					? (long)d
					: d)
			}";
		return v.ToString() ?? "";
	}
}