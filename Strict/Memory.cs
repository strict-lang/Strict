using Strict.Expressions;

namespace Strict;

public sealed class Memory
{
	/// <summary>
	/// Array-backed register file — O(1) access with no hashing overhead.
	/// </summary>
	public RegisterFile Registers { get; } = new();
	/// <summary>
	/// Current variable scope; replaced via <see cref="VirtualMachine"/> call stack.
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
			throw new InvalidOperationException("Cannot add to non-list variable: " + key);
		collection.List.Items.Add(element);
	}

	public void AddToDictionary(string variableKey, ValueInstance keyToAddTo, ValueInstance value)
	{
		Frame.TryGet(variableKey, out var collection);
		if (collection.IsDictionary)
			collection.GetDictionaryItems()[keyToAddTo] = value;
	}
}