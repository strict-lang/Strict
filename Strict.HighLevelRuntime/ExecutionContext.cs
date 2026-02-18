using Strict.Language;
using System.Collections;
using System.Collections.Generic;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ExecutionContext(Type type, Method method)
{
	public Type Type { get; } = type;
	public Method Method { get; } = method;
	public ExecutionContext? Parent { get; init; }
	public ValueInstance? This { get; init; }
	public Dictionary<string, ValueInstance> Variables { get; } = new(StringComparer.Ordinal);

	public ValueInstance Get(string name) =>
		Find(name) ?? throw new VariableNotFound(name, Type, This);

	public ValueInstance? Find(string name)
	{
		if (Variables.TryGetValue(name, out var v))
			return v;
		if (This == null)
			return Parent?.Find(name);
    if (This.ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary } dictionaryType &&
			This.Value is IDictionary dictionaryValues)
		{
			var member = Type.FindMember(name);
			if (member != null && IsListMemberType(member.Type))
			{
				var pairs = BuildDictionaryPairsList(dictionaryValues, member.Type,
					dictionaryType.ImplementationTypes[0], dictionaryType.ImplementationTypes[1]);
				return new ValueInstance(member.Type, pairs);
			}
		}
		if (This.Value is IDictionary<string, object?> rawMembers &&
			rawMembers.TryGetValue(name, out var rawValue))
		{
			var memberType = Type.FindMember(name)?.Type;
			if (memberType != null)
				return new ValueInstance(memberType, rawValue);
		}
		if (name == Type.ValueLowercase)
			return This;
		var implicitMember =
			Type.Members.FirstOrDefault(m => !m.IsConstant && m.Type.Name != Base.Iterator);
		if (implicitMember != null &&
			implicitMember.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
			return new ValueInstance(implicitMember.Type, This.Value);
		return Parent?.Find(name);
	}

	public ValueInstance Set(string name, ValueInstance value) => Variables[name] = value;

	internal static IList BuildDictionaryPairsList(IDictionary dictionaryValues, Type listMemberType,
		Type keyType, Type valueType)
	{
		var elementType = listMemberType is GenericTypeImplementation { Generic.Name: Base.List } listType
			? listType.ImplementationTypes[0]
			: listMemberType;
		var pairs = new List<ValueInstance>(dictionaryValues.Count);
		foreach (DictionaryEntry entry in dictionaryValues)
		{
			var keyInstance = new ValueInstance(keyType, entry.Key);
			var valueInstance = new ValueInstance(valueType, entry.Value);
			pairs.Add(new ValueInstance(elementType,
				new List<ValueInstance> { keyInstance, valueInstance }));
		}
		return pairs;
	}

	private static bool IsListMemberType(Type memberType) =>
		memberType.Name == Base.List ||
		memberType is GenericTypeImplementation { Generic.Name: Base.List };

	public sealed class VariableNotFound(string name, Type type, ValueInstance? instance)
		: Exception($"Variable '{name}' or member '{name}' of this type '{type}'" + (instance != null
			? $" (instance='{instance}')"
			: "") + " (or its parents) not found");

	public override string ToString() =>
		nameof(ExecutionContext) + " Type=" + Type.Name + ", This=" + This + ", Variables:" +
		Environment.NewLine + "  " + Variables.DictionaryToWordList(Environment.NewLine + "  ", " ",
			true);
}