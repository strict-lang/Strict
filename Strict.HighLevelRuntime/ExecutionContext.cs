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
			: Parent?.Get(name) ?? GetFromThisOrThrow(name);

	private ValueInstance GetFromThisOrThrow(string name)
	{
		if (This == null)
			throw new VariableNotFound(name);
		if (This.Value is IDictionary<string, object?> rawMembers)
		{
			if (!rawMembers.TryGetValue(name, out var rawValue))
				throw new VariableOrMemberNotFound(name, This);
			var memberType = Type.FindMember(name)?.Type;
			return memberType != null
				? new ValueInstance(memberType, rawValue)
				: throw new VariableOrMemberNotFound(name, This);
		}
		if (name == Base.ValueLowercase)
			return This;
		var implicitMember = Type.Members.FirstOrDefault(m => !m.IsConstant && m.Type.Name != Base.Iterator);
		if (implicitMember != null && implicitMember.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
			return new ValueInstance(implicitMember.Type, This.Value);
		throw new VariableOrMemberNotFound(name, This);
	}

	public ValueInstance Set(string name, ValueInstance value) => Variables[name] = value;

	public sealed class VariableOrMemberNotFound(string name, ValueInstance instance)
		: Exception($"Variable '{name}' or member '{name}' of this type '{instance}' not found");

	public sealed class VariableNotFound(string name) : Exception($"Variable '{name}' not found");

	public override string ToString() =>
		nameof(ExecutionContext) + " Type=" + Type.Name + ", This=" + This + ", Variables=" +
		Variables.DictionaryToWordList();
}