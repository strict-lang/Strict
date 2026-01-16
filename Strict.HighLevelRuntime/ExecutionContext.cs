using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ExecutionContext(Type type)
{
	public Type Type { get; } = type;
	public ExecutionContext? Parent { get; init; }
	public ValueInstance? This { get; init; }
	public Dictionary<string, ValueInstance> Variables { get; } = new(StringComparer.Ordinal);

	public ValueInstance Get(string name) =>
		Variables.TryGetValue(name, out var v)
			? v
			: Parent?.Get(name) ?? throw new VariableNotFound(name);

	public ValueInstance Set(string name, ValueInstance value) => Variables[name] = value;
	public sealed class VariableNotFound(string name) : Exception($"Variable '{name}' not found");

	public override string ToString() =>
		nameof(ExecutionContext) + " Type=" + Type.Name + ", This=" + This + ", Variables=" +
		Variables.ToWordList();
}