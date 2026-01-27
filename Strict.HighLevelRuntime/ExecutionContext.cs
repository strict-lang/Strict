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
			: Parent?.Get(name) ?? (This?.Value is IReadOnlyDictionary<string, ValueInstance> members
				? members.TryGetValue(name, out var member)
					? member
					: throw new VariableOrMemberNotFound(name, This)
				: This) ?? throw new VariableNotFound(name);

	public ValueInstance Set(string name, ValueInstance value) => Variables[name] = value;

	public sealed class VariableOrMemberNotFound(string name, ValueInstance instance)
		: Exception($"Variable '{name}' or member '{name}' of this type '{instance}' not found");

	public sealed class VariableNotFound(string name) : Exception($"Variable '{name}' not found");

	public override string ToString() =>
		nameof(ExecutionContext) + " Type=" + Type.Name + ", This=" + This + ", Variables=" +
		Variables.DictionaryToWordList();
}