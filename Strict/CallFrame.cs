using Strict.Expressions;
using Strict.Language;

namespace Strict;

/// <summary>
/// Variable scope for one method invocation. Writes always go to this frame's own lazy dict;
/// reads walk the parent chain but only through member variables, so child calls can access
/// the caller's 'has' fields without copying them. Modeled after ExecutionContext.
/// </summary>
internal sealed class CallFrame(CallFrame? parent = null)
{
	public CallFrame(IReadOnlyDictionary<string, ValueInstance>? initialVariables) : this()
	{
		if (initialVariables != null)
			foreach (var (name, value) in initialVariables)
				Set(name, value);
	}

	private Dictionary<string, ValueInstance>? variables;
	private HashSet<string>? memberNames;
	/// <summary>
	/// Materialized locals dict — used by <see cref="Memory.Variables"/> for test compatibility
	/// </summary>
	internal Dictionary<string, ValueInstance> Variables =>
		variables ??= new Dictionary<string, ValueInstance>();

	internal bool TryGet(string name, out ValueInstance value)
	{
		if (variables != null && variables.TryGetValue(name, out value))
			return true;
		if (parent != null)
			return parent.TryGetMember(name, out value);
		value = default;
		return false;
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

	public override string ToString() =>
		nameof(CallFrame) + " " + nameof(variables) + ": " + variables?.DictionaryToWordList() +
		", members: " + (memberNames != null
			? string.Join(", ", memberNames)
			: "") +
		(parent != null
			? "\n\tParent: " + parent
			: "");
}