using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

public sealed class Memory
{
	public Dictionary<string, Instance> Variables = new();
	public Dictionary<Register, Instance> Registers { get; init; } = new();

	public void AddToCollectionVariable(string key, object element)
	{
		Variables.TryGetValue(key, out var collection);
		if (collection?.Value is List<Expression> listExpression)
			listExpression.Add(ConvertObjectToValueForm(element, listExpression[0]));
	}

	public void AddToDictionary(string variableKey, object keyToAddTo, object value)
	{
		Variables.TryGetValue(variableKey, out var collection);
		if (collection?.Value is not Dictionary<Type, Type> dictionary)
			return;
		var keyType = ((Instance)keyToAddTo).ReturnType;
		var valueType = ((Instance)value).ReturnType;
		if (keyType == null || valueType == null)
			return;
		dictionary.Add(keyType, valueType);
	}

	private static Value ConvertObjectToValueForm(object obj, Expression prototype) =>
		new(prototype.ReturnType, obj);
}