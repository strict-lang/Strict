using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class Memory
{
	public Dictionary<string, Instance> Variables = new();
	public Dictionary<Register, Instance> Registers { get; init; } = new();

	public void AddToCollectionVariable(string key, object element)
	{
		Variables.TryGetValue(key, out var collection);
		if (collection?.Value is List<Expression> listExpression)
			listExpression.Add(listExpression.Count > 0 ? ConvertObjectToValueForm(element, listExpression[0]) : ConvertToValueFormWhenListIsEmpty(collection, element));
	}

	public void AddToDictionary(string variableKey, Instance keyToAddTo, Instance value)
	{
		Variables.TryGetValue(variableKey, out var collection);
		if (collection?.Value is not Dictionary<Value, Value> dictionary ||
			keyToAddTo.ReturnType == null || value.ReturnType == null)
			return;
		dictionary.Add(new Value(keyToAddTo.ReturnType, keyToAddTo.Value),
			new Value(value.ReturnType, value.Value));
	}

	private static Value ConvertObjectToValueForm(object obj, Expression prototype) =>
		new(prototype.ReturnType, obj);
	private static Value ConvertToValueFormWhenListIsEmpty(Instance collection, object element) =>
		collection.ReturnType is not GenericTypeImplementation genericImplementationType
			? throw new InvalidOperationException()
			: new Value(genericImplementationType.ImplementationTypes[0], element);
}