using Strict.Expressions;

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
			return; //TODO: shouldn't this be an error?
		//TODO: this seems all very fishy, if we have a mutable list, why do we need to make copies all the time?
		var listItems = new ValueInstance[collection.List.Items.Length + 1];
		collection.List.Items.CopyTo(listItems);
		listItems[^1] = element;
		Frame.Set(key, new ValueInstance(collection.List.ReturnType, listItems));
		/*TODO: can't see how this is still needed? clean up!
			return;
		}
		if (collection.GetTypeExceptText() is not GenericTypeImplementation genericImplementationType)
			throw new InvalidOperationException();
		listItems.Add(element);
		Frame.Set(key, new ValueInstance(genericImplementationType, listItems));
		*/
	}

	public void AddToDictionary(string variableKey, ValueInstance keyToAddTo, ValueInstance value)
	{
		Frame.TryGet(variableKey, out var collection);
		if (collection.IsDictionary)
			//TODO: same here, why not modify the dictionary?
			Frame.Set(variableKey, new ValueInstance(collection.GetTypeExceptText(),
				new Dictionary<ValueInstance, ValueInstance>(collection.GetDictionaryItems())
				{
					{ keyToAddTo, value }
				}));
	}
}