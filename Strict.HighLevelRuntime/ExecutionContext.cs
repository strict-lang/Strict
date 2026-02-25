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
	public Dictionary<string, ValueInstance> Variables { get; } = new(StringComparer.Ordinal);
	public ValueInstance? ExitMethodAndReturnValue { get; internal set; }

	public ValueInstance Get(string name) =>
		Find(name) ?? throw new VariableNotFound(name, Type, This);

	public ValueInstance? Find(string name)
	{
		if (Variables.TryGetValue(name, out var v))
			return v;
		if (This == null)
			return Parent?.Find(name);
		if (name == Type.ValueLowercase)
			return This;
		var implicitMember =
			Type.Members.FirstOrDefault(m => !m.IsConstant && m.Type.Name != Base.Iterator);
		if (implicitMember != null &&
			implicitMember.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
			return new ValueInstance(implicitMember.Type, This.Value);
		return Parent?.Find(name);
	}

	public ValueInstance Set(string name, ValueInstance value)
	{
		var ctx = this;
		while (ctx != null)
		{
			if (ctx.Variables.ContainsKey(name))
				return ctx.Variables[name] = value;
			ctx = ctx.Parent;
		}
		return Variables[name] = value;
	}

	internal static IList BuildDictionaryPairsList(Type listMemberType,
		Dictionary<ValueInstance, ValueInstance> dictionary)
	{
		var elementType = listMemberType is GenericTypeImplementation { Generic.Name: Base.List } list
			? list.ImplementationTypes[0]
			: listMemberType;
		var pairs = new List<ValueInstance>(dictionary.Count);
		foreach (var entry in dictionary)
			pairs.Add(new ValueInstance(elementType, new List<ValueInstance> { entry.Key, entry.Value }));
		return pairs;
	}

	public sealed class VariableNotFound(string name, Type type, ValueInstance? instance)
		: Exception($"Variable '{name}' or member '{name}' of this type '{type}'" + (instance != null
			? $" (instance='{instance}')"
			: "") + " (or its parents) not found");

	public override string ToString() =>
		nameof(ExecutionContext) + " Type=" + Type.Name + ", This=" + This + ", Variables:" +
		Environment.NewLine + "  " +
		Variables.DictionaryToWordList(Environment.NewLine + "  ", " ", true);
}