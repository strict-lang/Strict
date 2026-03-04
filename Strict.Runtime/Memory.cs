using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

public sealed class Memory
{
	public void AddToCollectionVariable(string key, object element)
	{
		Variables.TryGetValue(key, out var collection);
		if (collection?.Value is List<Expression> listExpression)
			listExpression.Add(listExpression.Count > 0
				? ConvertObjectToValueForm(element, listExpression[0])
				: ConvertToValueFormWhenListIsEmpty(collection, element));
	}

	public Dictionary<string, Instance> Variables = new();
	public Dictionary<Register, Instance> Registers { get; init; } = new();
	/// <summary>Names of variables that belong to the calling instance's members.</summary>
	public HashSet<string> MemberVariableNames { get; } = [];

	public void AddToDictionary(string variableKey, Instance keyToAddTo, Instance value)
	{
		Variables.TryGetValue(variableKey, out var collection);
		if (collection?.Value is Dictionary<Value, Value> dictionary)
			dictionary.Add(
				new Value(keyToAddTo.ReturnType, ToValueInstance(keyToAddTo.ReturnType, keyToAddTo.Value)),
				new Value(value.ReturnType, ToValueInstance(value.ReturnType, value.Value)));
	}

	private static ValueInstance ToValueInstance(Type type, object obj)
	{
		if (obj is Instance inst)
			obj = inst.Value;
		if (obj is ValueInstance vi)
			return vi;
		if (type.IsNumber)
			return new ValueInstance(type, Convert.ToDouble(obj));
		if (type.IsText)
			return new ValueInstance(obj?.ToString() ?? "");
		return new ValueInstance(type, Convert.ToDouble(obj));
	}

	private static Value ConvertObjectToValueForm(object obj, Expression prototype) =>
		new(prototype.ReturnType, ToValueInstance(prototype.ReturnType, obj));

	private static Value ConvertToValueFormWhenListIsEmpty(Instance collection, object element) =>
		collection.ReturnType is not GenericTypeImplementation genericImplementationType
			? throw new InvalidOperationException()
			: new Value(genericImplementationType.ImplementationTypes[0],
				ToValueInstance(genericImplementationType.ImplementationTypes[0], element));
}