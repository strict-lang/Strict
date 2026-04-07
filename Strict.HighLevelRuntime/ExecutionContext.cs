using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ExecutionContext(Type type, Method method, ValueInstance? thisInstance = null,
	ExecutionContext? parent = null)
{
	public Type Type { get; private set; } = type;
	public Method Method { get; private set; } = method;
	public ExecutionContext? Parent { get; private set; } = parent;
	public ValueInstance? This { get; private set; } = thisInstance;
	public bool IsTestAtCurrentLine { get; set; }
	private Dictionary<string, ValueInstance>? variables;
	/// <summary>
	/// Lazy-initialized: only created when a variable is actually written, avoiding allocation for
	/// methods that never declare local variables (the majority of test-runner invocations).
	/// </summary>
	public Dictionary<string, ValueInstance> Variables =>
		variables ??= new Dictionary<string, ValueInstance>(StringComparer.Ordinal);
	public ValueInstance? ExitMethodAndReturnValue { get; internal set; }
	public int CurrentExpressionLineNumber { get; internal set; } = -1;

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
		var members = Type.Members;
		for (var i = 0; i < members.Count; i++)
			if (!members[i].IsConstant && members[i].Type.Name != Type.Iterator &&
				members[i].Name.Equals(name, StringComparison.OrdinalIgnoreCase))
				return new ValueInstance(This.Value, members[i].Type);
		return Parent?.Find(name, statistics);
	}

	/// <summary>
	/// Clears local variables and resets the early-exit so the same context can be reused in the
	/// next for iteration, saving <see cref="Dictionary{TKey, TValue}"/> and context allocations.
	/// </summary>
	public void ResetIteration()
	{
		variables?.Clear();
		ExitMethodAndReturnValue = null;
		CurrentExpressionLineNumber = -1;
	}

	/// <summary>
	/// Full reset for pool reuse: updates all identity fields and clears mutable state.
	/// </summary>
	internal void Reset(Type newType, Method newMethod, ValueInstance? instance,
		ExecutionContext? newParent)
	{
		Type = newType;
		Method = newMethod;
		This = instance;
		Parent = newParent;
		variables?.Clear();
		ExitMethodAndReturnValue = null;
		CurrentExpressionLineNumber = -1;
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

	public sealed class VariableNotFound(string name, Type type, ValueInstance? instance)
		: Exception($"Variable '{name}' or member '{name}' of this type '{type}'" + (instance != null
			? $" (instance='{instance}')"
			: "") + " (or its parents) not found");

	public override string ToString() =>
		nameof(ExecutionContext) + " Type=" + Type.Name + ", This=" + This + ", Variables:" +
		Environment.NewLine + "  " +
		(variables?.DictionaryToWordList(Environment.NewLine + "  ", " ", true) ?? "");
}