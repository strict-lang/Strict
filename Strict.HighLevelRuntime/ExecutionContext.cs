using Strict.Expressions;
using Strict.Language;
using System.Collections;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ExecutionContext(Type type, Method method)
{
	public Type Type { get; } = type;
	public Method Method { get; } = method;
	public ExecutionContext? Parent { get; init; }
	public ValueInstance? This { get; init; }
	private Dictionary<string, ValueInstance>? variables;
	/// <summary>
	/// Lazy-initialized: only created when a variable is actually written, avoiding allocation for
	/// methods that never declare local variables (the majority of test-runner invocations).
	/// </summary>
	public Dictionary<string, ValueInstance> Variables => variables ??= new(StringComparer.Ordinal);
	public ValueInstance? ExitMethodAndReturnValue { get; internal set; }

	public ValueInstance Get(string name, Statistics statistics) =>
		Find(name, statistics) ?? throw new VariableNotFound(name, Type, This);

	public ValueInstance? Find(string name, Statistics statistics)
	{
		statistics.FindVariableCount++;
		if (variables != null && variables.TryGetValue(name, out var v))
			return v;
		if (This == null)
			return Parent?.Find(name, statistics);
		if (name == Type.ValueLowercase)
			return This;
		var implicitMember =
			Type.Members.FirstOrDefault(m => !m.IsConstant && m.Type.Name != Type.Iterator);
		if (implicitMember != null &&
			implicitMember.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
			return new ValueInstance(This.Value, implicitMember.Type);
		return Parent?.Find(name, statistics);
	}

	/// <summary>
	/// Clears local variables and resets the early-exit flag so the same context can be reused
	/// across for-loop iterations, saving one <see cref="Dictionary{TKey, TValue}"/> + one
	/// <see cref="ExecutionContext"/> allocation per iteration.
	/// Calling this on a freshly-created context (before the first iteration) is safe and a no-op:
	/// <c>variables?.Clear()</c> does nothing when the dictionary has not yet been allocated.
	/// Using <c>Clear()</c> rather than setting <c>variables = null</c> is intentional: it keeps
	/// the already-allocated dictionary alive so it can be reused without re-allocating.
	/// </summary>
	public void ResetForLoopIteration()
	{
		variables?.Clear();
		ExitMethodAndReturnValue = null;
	}

	public ValueInstance Set(string name, ValueInstance value)
	{
		var ctx = this;
		while (ctx != null)
		{
			if (ctx.variables != null && ctx.variables.ContainsKey(name))
				return ctx.variables[name] = value;
			ctx = ctx.Parent;
		}
		return Variables[name] = value;
	}

	internal static IList BuildDictionaryPairsList(Type listMemberType,
		Dictionary<ValueInstance, ValueInstance> dictionary)
	{
		var elementType = listMemberType is GenericTypeImplementation { Generic.Name: Type.List } list
			? list.ImplementationTypes[0]
			: listMemberType;
		var pairs = new List<ValueInstance>(dictionary.Count);
		foreach (var entry in dictionary)
			pairs.Add(new ValueInstance(elementType,
				new List<ValueInstance> { entry.Key, entry.Value }));
		return pairs;
	}

	public sealed class VariableNotFound(string name, Type type, ValueInstance? instance)
		: Exception($"Variable '{name}' or member '{name}' of this type '{type}'" + (instance != null
			? $" (instance='{instance}')"
			: "") + " (or its parents) not found");

	public override string ToString() =>
		nameof(ExecutionContext) + " Type=" + Type.Name + ", This=" + This + ", Variables:" +
		Environment.NewLine + "  " +
		(variables?.DictionaryToWordList(Environment.NewLine + "  ", " ", true) ?? "");

	/*TODO: probably eats up memory! avoid!
	public void AddDictionaryElements(ValueInstance? instance)
	{
		if (instance?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Type.Dictionary
			} implementation ||
			instance.Value.Value is not Dictionary<ValueInstance, ValueInstance> dictionary ||
			Variables.ContainsKey(Type.ElementsLowercase))
			return;
		var listMemberType = implementation.Members.FirstOrDefault(member =>
			member.Type is GenericTypeImplementation { Generic.Name: Type.List } ||
			member.Type.IsList)?     .Type ?? implementation.GetType(Type.List);
		var listValue = ExecutionContext.BuildDictionaryPairsList(listMemberType, dictionary);
		Set(Type.ElementsLowercase, ValueInstance.CreateObject(listMemberType, listValue));
	}
	*/
}