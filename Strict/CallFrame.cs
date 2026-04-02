using Strict.Expressions;
using Strict.Language;

namespace Strict;

/// <summary>
/// Variable scope for one method invocation. Writes always go to this frame's own lazy dict;
/// reads walk the parent chain but only through member variables, so child calls can access
/// the caller's 'has' fields without copying them. Modeled after ExecutionContext.
/// </summary>
internal sealed class CallFrame
{
	internal CallFrame(CallFrame? parent = null) => this.parent = parent;

	public CallFrame(IReadOnlyDictionary<string, ValueInstance>? initialVariables)
	{
		if (initialVariables != null)
			foreach (var (name, value) in initialVariables)
				Set(name, value);
	}

	private CallFrame? parent;
	private Dictionary<string, ValueInstance>? variables;
	private HashSet<string>? memberNames;
	/// <summary>
	/// Materialized locals dict — used by <see cref="Memory.Variables"/> for test compatibility
	/// </summary>
	internal Dictionary<string, ValueInstance> Variables =>
		variables ??= new Dictionary<string, ValueInstance>();

	//TODO: called 4 million times, needs to be avoided. it doesn't even make sense to call this that often, we have many 5 lookups in AdjustBrightness, rest is just repeating the same stuff over and over
	//TODO: main optimization in the for loop is to take the image.Colors(colorIndex) and to work directly on it without looking it up multiple times per iteration. this is still a VirtualMachine, but we don't have to be stupid!
	internal bool TryGet(string name, out ValueInstance value)
	{
		if (variables != null && variables.TryGetValue(name, out value))
			return true; //80% ends up here
		if (parent != null && parent.TryGetMember(name, out value))
			return true; //15% here
		var dotIndex = name.IndexOf('.');
		if (dotIndex > 0 && TryGetRootValue(name, dotIndex, out var root) &&
			TryGetNestedMemberValue(root, name, dotIndex + 1, out value))
			return true;
		value = default;
		return false;
	}

	private bool TryGetRootValue(string name, int dotIndex, out ValueInstance value)
	{
		var rootName = name[..dotIndex];
		if (variables != null && variables.TryGetValue(rootName, out value))
			return true;
		if (parent != null && parent.TryGetMember(rootName, out value))
			return true;
		value = default;
		return false;
	}

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

	private bool TryGetMember(string name, out ValueInstance value)
	{
		if (memberNames != null && memberNames.Contains(name) &&
			variables != null && variables.TryGetValue(name, out value))
			return true;
		value = default;
		return false;
	}

	internal ValueInstance Get(string name) =>
		TryGet(name, out var value)
			? value
			: throw new ValueNotFound(name, this);

	private sealed class ValueNotFound(string message, CallFrame frame)
		: Exception(message + " in " + frame);

	/// <summary>
	/// Always writes to this frame's own dict (never clobbers parent).
	/// </summary>
	internal void Set(string name, ValueInstance value, bool isMember = false)
	{
		variables ??= new Dictionary<string, ValueInstance>();
		variables[name] = value;
		if (isMember)
		{
			memberNames ??= [];
			memberNames.Add(name);
		}
	}

	internal void Clear()
	{
		variables?.Clear();
		memberNames?.Clear();
	}

	/// <summary>
	/// Resets this frame for reuse from the pool: clears all variables and sets a new parent.
	/// </summary>
	internal void Reset(CallFrame? newParent)
	{
		variables?.Clear();
		memberNames?.Clear();
		parent = newParent;
	}

	public override string ToString() =>
		nameof(CallFrame) + " " + nameof(variables) + ": " + variables?.DictionaryToWordList() +
		", members: " + (memberNames != null
			? string.Join(", ", memberNames)
			: "") +
		(parent != null
			? "\n\tParent: " + parent
			: "");
}