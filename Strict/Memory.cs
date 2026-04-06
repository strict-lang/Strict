using Strict.Expressions;
using Strict.Language;

namespace Strict;

public sealed class Memory
{
	private readonly RegisterFile registers = new();
	/// <summary>
  /// Array-backed register file — O(1) access with no hashing overhead.
	/// </summary>
	public RegisterFile Registers
	{
		get
		{
			if (PerformanceLog.IsEnabled)
       PerformanceLog.Write("Memory.Registers get", "callers=" + PerformanceLog.GetCallers(1));
			return registers;
		}
	}
	private CallFrame frame = new();
	/// <summary>
 /// Current variable scope; replaced via <see cref="VirtualMachine"/> call stack.
	/// </summary>
	internal CallFrame Frame
	{
		get
		{
			if (PerformanceLog.IsEnabled)
				PerformanceLog.Write("Memory.Frame get", "access");
			return frame;
		}
		set
		{
			if (PerformanceLog.IsEnabled)
				PerformanceLog.Write("Memory.Frame set", "frame=" + value.GetHashCode());
			frame = value;
		}
	}
	/// <summary>
  /// Exposes the current frame's local variable dict for backward-compatible test access.
	/// Use <see cref="Frame"/> methods for scoped lookup inside the interpreter.
	/// </summary>
	public Dictionary<string, ValueInstance> Variables
	{
		get
		{
			if (PerformanceLog.IsEnabled)
				PerformanceLog.Write("Memory.Variables get", "access");
			return Frame.Variables;
		}
	}

	public void AddToCollection(int symbolId, ValueInstance element)
	{
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("Memory.AddToCollection", "key=" + CallFrame.GetSymbolName(symbolId) +
				", element=" + Describe(element));
		Frame.TryGet(symbolId, out var collection);
		if (!collection.IsList)
			throw new InvalidOperationException("Cannot add to non-list variable \"" +
				CallFrame.GetSymbolName(symbolId) + "\" of type " + collection.GetType().Name);
		collection.List.Items.Add(element);
	}

 public void AddToCollection(string key, ValueInstance element)
	{
		var symbolId = CallFrame.ResolveSymbolId(key);
		if (Frame.TryGet(symbolId, out _))
		{
			AddToCollection(symbolId, element);
			return;
		}
    var hasCollection = Variables.TryGetValue(key, out var collection);
		if (!hasCollection || !collection.IsList)
			throw new InvalidOperationException("Cannot add to non-list variable \"" + key +
       "\" of type " + (hasCollection ? collection.GetType().Name : "unset"));
		collection.List.Items.Add(element);
	}

	public void AddToDictionary(int symbolId, ValueInstance keyToAddTo, ValueInstance value)
	{
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("Memory.AddToDictionary", "key=" + CallFrame.GetSymbolName(symbolId) +
				", itemKey=" + Describe(keyToAddTo) + ", value=" + Describe(value));
		Frame.TryGet(symbolId, out var collection);
		if (collection.IsDictionary)
			collection.GetDictionaryItems()[keyToAddTo] = value;
	}

	public void AddToDictionary(string variableKey, ValueInstance keyToAddTo, ValueInstance value) =>
		AddToDictionary(CallFrame.ResolveSymbolId(variableKey), keyToAddTo, value);

	private static string Describe(ValueInstance value)
	{
		if (!value.HasValue)
			return "unset";
		if (value.IsText)
			return "Text(length=" + value.Text.Length + ")";
		if (value.IsList)
			return "List(type=" + value.List.ReturnType.Name + ", count=" + value.List.Items.Count + ")";
		if (value.IsDictionary)
			return "Dictionary(count=" + value.GetDictionaryItems().Count + ")";
		var typeInstance = value.TryGetValueTypeInstance();
		return typeInstance != null
			? "TypeInstance(type=" + typeInstance.ReturnType.Name + ", members=" + typeInstance.Values.Length + ")"
			: value.GetType().Name + "(" + value.Number + ")";
	}
}