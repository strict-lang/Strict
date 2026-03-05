using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

public sealed class Memory
{
	public void AddToCollectionVariable(string key, ValueInstance element)
	{
		Variables.TryGetValue(key, out var collection);
		if (!collection.IsList)
			return;
		var listItems = new List<ValueInstance>(collection.List.Items);
		if (listItems.Count > 0)
		{
			listItems.Add(element);
			Variables[key] = new ValueInstance(collection.List.ReturnType, listItems);
			return;
		}
		if (collection.GetTypeExceptText() is not GenericTypeImplementation genericImplementationType)
			throw new InvalidOperationException();
		listItems.Add(element);
		Variables[key] = new ValueInstance(genericImplementationType, listItems);
	}

	public Dictionary<string, ValueInstance> Variables = new();
	public Dictionary<Register, ValueInstance> Registers { get; init; } = new();
	/// <summary>Names of variables that belong to the calling instance's members.</summary>
	public HashSet<string> MemberVariableNames { get; } = [];

	public void AddToDictionary(string variableKey, ValueInstance keyToAddTo, ValueInstance value)
	{
		Variables.TryGetValue(variableKey, out var collection);
		if (!collection.IsDictionary)
			return;
		var dictionaryItems = new Dictionary<ValueInstance, ValueInstance>(collection.GetDictionaryItems());
		dictionaryItems.Add(keyToAddTo, value);
		Variables[variableKey] = new ValueInstance(collection.GetTypeExceptText(), dictionaryItems);
	}
}