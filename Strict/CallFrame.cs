using Strict.Expressions;
using Strict.Language;
using System.Reflection.Metadata;
using Type = Strict.Language.Type;

namespace Strict;

/// <summary>
/// Variable scope for one method invocation. Writes always go to this frame's own lazy dict;
/// reads walk the parent chain but only through member variables, so child calls can access
/// the caller's 'has' fields without copying them. Modeled after ExecutionContext.
/// </summary>
internal sealed class CallFrame
{
	internal CallFrame(CallFrame? parent = null) => this.parent = parent;
  private static readonly Lock symbolLock = new();
	private static readonly Dictionary<string, int> symbolIds = new(StringComparer.Ordinal);
	private static readonly List<string> symbolNames = [];
	internal static readonly int ValueSymbolId = ResolveSymbolId(Type.ValueLowercase);
	internal static readonly int IndexSymbolId = ResolveSymbolId(Type.IndexLowercase);
	internal static readonly int OuterSymbolId = ResolveSymbolId(Type.OuterLowercase);
	internal static readonly int ElementsSymbolId = ResolveSymbolId(Type.ElementsLowercase);
	internal static readonly int CharactersSymbolId = ResolveSymbolId("characters");

	public CallFrame(IReadOnlyDictionary<string, ValueInstance>? initialVariables)
	{
		if (initialVariables != null)
			foreach (var (name, value) in initialVariables)
				Set(name, value);
	}

	private CallFrame? parent;
	private Dictionary<string, ValueInstance>? variables;
  private ValueInstance[] slots = [];
	private bool[] memberSlots = [];
	private int highestAssignedSymbolId = -1;
	/// <summary>
	/// Materialized locals dict — used by <see cref="Memory.Variables"/> for test compatibility
	/// </summary>
 internal Dictionary<string, ValueInstance> Variables
	{
		get
		{
			if (PerformanceLog.IsEnabled)
				PerformanceLog.Write("CallFrame.Variables", "access");
     return variables ??= CreateVariablesSnapshot();
		}
	}

	private Dictionary<string, ValueInstance> CreateVariablesSnapshot()
	{
		var snapshot = new Dictionary<string, ValueInstance>(Math.Max(highestAssignedSymbolId + 1, 0));
		for (var symbolId = 0; symbolId <= highestAssignedSymbolId; symbolId++)
			if (symbolId < slots.Length && slots[symbolId].HasValue)
				snapshot[GetSymbolName(symbolId)] = slots[symbolId];
		return snapshot;
	}

	internal static int ResolveSymbolId(string name)
	{
		lock (symbolLock)
		{
			if (symbolIds.TryGetValue(name, out var symbolId))
				return symbolId;
			symbolId = symbolNames.Count;
			symbolIds.Add(name, symbolId);
			symbolNames.Add(name);
			return symbolId;
		}
	}

	internal static string GetSymbolName(int symbolId)
	{
		lock (symbolLock)
			return symbolId >= 0 && symbolId < symbolNames.Count
				? symbolNames[symbolId]
				: symbolId.ToString();
	}

	//TODO: called 4 million times, needs to be avoided. it doesn't even make sense to call this that often, we have many 5 lookups in AdjustBrightness, rest is just repeating the same stuff over and over
	//TODO: main optimization in the for loop is to take the image.Colors(colorIndex) and to work directly on it without looking it up multiple times per iteration. this is still a VirtualMachine, but we don't have to be stupid!
	internal bool TryGet(string name, out ValueInstance value)
	{
   if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("CallFrame.TryGet", "name=" + name);
		var dotIndex = name.IndexOf('.');
    return dotIndex <= 0
			? TryGet(ResolveSymbolId(name), out value)
			: TryGetNestedPath(name, dotIndex, out value);
	}

 internal bool TryGet(int symbolId, out ValueInstance value)
	{
   if (TryGetSlotValue(symbolId, false, out value))
			return true;
		if (parent != null && parent.TryGetMember(symbolId, out value))
			return true;
		value = default;
		return false;
	}

  private bool TryGetNestedPath(string name, int dotIndex, out ValueInstance value)
	{
   if (TryGet(ResolveSymbolId(name[..dotIndex]), out var root) &&
			TryGetNestedMemberValue(root, name, dotIndex + 1, out value))
			return true;
		value = default;
		return false;
	}

	//TODO: never called
	private static bool TryGetNestedMemberValue(ValueInstance root, string path, int segmentStart,
		out ValueInstance value)
	{
		var current = root;
		while (true)
		{
			var currentTypeInstance = current.TryGetValueTypeInstance();
			if (currentTypeInstance == null)
			{
				value = default;
				return false;
			}
			var nextDotIndex = path.IndexOf('.', segmentStart);
			var segment = nextDotIndex < 0
				? path[segmentStart..]
				: path[segmentStart..nextDotIndex];
			if (!currentTypeInstance.TryGetValue(segment, out current))
			{
				value = default;
				return false;
			}
			if (nextDotIndex < 0)
			{
				value = current;
				return true;
			}
			segmentStart = nextDotIndex + 1;
		}
	}

	//TODO: slow and complicated, but only called for member lookups
 private bool TryGetMember(int symbolId, out ValueInstance value)
	{
   if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("CallFrame.TryGetMember", "name=" + GetSymbolName(symbolId));
		return TryGetSlotValue(symbolId, true, out value);
	}

	private bool TryGetSlotValue(int symbolId, bool requireMember, out ValueInstance value)
	{
		if (!requireMember && symbolId == OuterSymbolId && parent != null &&
			parent.TryGet(ValueSymbolId, out value))
			return true;
   if (symbolId >= 0 && symbolId < slots.Length && slots[symbolId].HasValue &&
			(!requireMember || symbolId < memberSlots.Length && memberSlots[symbolId]))
		{
			value = slots[symbolId];
			if (value.IsText && value.Text.StartsWith("for elements"))
				throw new NotSupportedException("Invalid CallFrame.TryGet variable: " +
					GetSymbolName(symbolId) + " " + value.Text);
			return true;
		}
		value = default;
		return false;
	}

	//TODO: we shouldn't have these many ways of getting a variable
 internal ValueInstance Get(string name)
	{
   if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("CallFrame.Get", "name=" + name);
    return TryGet(name, out var value)
			? value
			: throw new ValueNotFound(name, this);
	}

	internal ValueInstance Get(int symbolId)
	{
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("CallFrame.Get", "name=" + GetSymbolName(symbolId));
		return TryGet(symbolId, out var value)
			? value
      : throw new ValueNotFound(GetSymbolName(symbolId), this);
	}

	private sealed class ValueNotFound(string message, CallFrame frame)
		: Exception(message + " in " + frame);

	/// <summary>
	/// Always writes to this frame's own dict (never clobbers parent).
	/// </summary>
	internal void Set(string name, ValueInstance value, bool isMember = false)
   => Set(ResolveSymbolId(name), value, isMember, name);

	internal void Set(int symbolId, ValueInstance value, bool isMember = false,
		string? name = null)
	{
    if (PerformanceLog.IsEnabled)
      PerformanceLog.Write("CallFrame.Set", "name=" + (name ?? GetSymbolName(symbolId)) +
				", isMember=" + isMember + ", value=" + Describe(value));
		//TODO: remove
		if (value.IsText && value.Text.StartsWith("for elements"))
      throw new NotSupportedException("Invalid CallFrame.Set variable: isMember=" +
				isMember + ", name: " + (name ?? GetSymbolName(symbolId)) + ", value: " +
				value.Text);
		EnsureCapacity(symbolId);
		slots[symbolId] = value;
		memberSlots[symbolId] = isMember;
		highestAssignedSymbolId = Math.Max(highestAssignedSymbolId, symbolId);
		if (variables != null)
			variables[name ?? GetSymbolName(symbolId)] = value;
	}

	internal void Clear()
	{
   if (PerformanceLog.IsEnabled)
      PerformanceLog.Write("CallFrame.Clear", "locals=" + (variables?.Count ?? 0) + ", members=" + CountMembers());
		if (highestAssignedSymbolId >= 0)
		{
			Array.Clear(slots, 0, highestAssignedSymbolId + 1);
			Array.Clear(memberSlots, 0, highestAssignedSymbolId + 1);
			highestAssignedSymbolId = -1;
		}
		variables?.Clear();
	}

	/// <summary>
	/// Resets this frame for reuse from the pool: clears all variables and sets a new parent.
	/// </summary>
	internal void Reset(CallFrame? newParent)
	{
   if (PerformanceLog.IsEnabled)
      PerformanceLog.Write("CallFrame.Reset", "locals=" + (variables?.Count ?? 0) + ", members=" + CountMembers() + ", parent=" + (newParent != null));
		Clear();
		parent = newParent;
	}

	private void EnsureCapacity(int symbolId)
	{
		if (symbolId < slots.Length)
			return;
		var newLength = Math.Max(symbolId + 1, slots.Length == 0
			? 8
			: slots.Length * 2);
		Array.Resize(ref slots, newLength);
		Array.Resize(ref memberSlots, newLength);
	}

	private int CountMembers()
	{
		var count = 0;
		for (var symbolId = 0; symbolId <= highestAssignedSymbolId && symbolId < memberSlots.Length; symbolId++)
			if (memberSlots[symbolId] && slots[symbolId].HasValue)
				count++;
		return count;
	}

	//TODO: only in debug mode, only used when LogPerformance.IsEnabled
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

	public override string ToString() =>
		nameof(CallFrame) + " " + nameof(variables) + ": " + variables?.DictionaryToWordList() +
    ", members: " + string.Join(", ", Enumerable.Range(0, highestAssignedSymbolId + 1).
			Where(symbolId => symbolId < memberSlots.Length && memberSlots[symbolId] &&
				slots[symbolId].HasValue).Select(GetSymbolName)) +
		(parent != null
			? "\n\tParent: " + parent
			: "");
}