using Strict.Expressions;
using Strict.Language;

namespace Strict.Runtime;

public sealed class Memory
{
	/// <summary>
	/// Array-backed register file — O(1) access with no hashing overhead.
	/// </summary>
	public RegisterFile Registers { get; } = new();

	/// <summary>
	/// Current variable scope; replaced via <see cref="BytecodeInterpreter"/> call stack.
	/// </summary>
	internal CallFrame Frame { get; set; } = new();

	/// <summary>
	/// Exposes the current frame's local variable dict for backward-compatible test access.
	/// Use <see cref="Frame"/> methods for scoped lookup inside the interpreter.
	/// </summary>
	public Dictionary<string, ValueInstance> Variables => Frame.Variables;

	public void AddToCollectionVariable(string key, ValueInstance element)
	{
		Frame.TryGet(key, out var collection);
		if (!collection.IsList)
			return;
		var listItems = new List<ValueInstance>(collection.List.Items);
		if (listItems.Count > 0)
		{
			listItems.Add(element);
			Frame.Set(key, new ValueInstance(collection.List.ReturnType, listItems));
			return;
		}
		if (collection.GetTypeExceptText() is not GenericTypeImplementation genericImplementationType)
			throw new InvalidOperationException();
		listItems.Add(element);
		Frame.Set(key, new ValueInstance(genericImplementationType, listItems));
	}

	public void AddToDictionary(string variableKey, ValueInstance keyToAddTo, ValueInstance value)
	{
		Frame.TryGet(variableKey, out var collection);
		if (collection.IsDictionary)
			Frame.Set(variableKey, new ValueInstance(collection.GetTypeExceptText(),
				new Dictionary<ValueInstance, ValueInstance>(collection.GetDictionaryItems())
				{
					{ keyToAddTo, value }
				}));
	}
}