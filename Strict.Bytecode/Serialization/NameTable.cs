using System.Collections;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// Used in BytecodeSerializer to write out strings, names, identifiers, etc. once.
/// </summary>
public sealed class NameTable : IEnumerable<string>
{
	public NameTable() { }

	public NameTable(BinaryReader reader)
	{
		var count = reader.Read7BitEncodedInt();
		for (var index = 0; index < count; index++)
			Add(reader.ReadString());
	}

	public NameTable CollectStrings(Instruction instruction) =>
		instruction switch
		{
			LoadVariableToRegister loadVar => Add(loadVar.Identifier),
			StoreVariableInstruction storeVar => Add(storeVar.Identifier).
				CollectValueInstanceStrings(storeVar.ValueInstance),
			StoreFromRegisterInstruction storeReg => Add(storeReg.Identifier),
			SetInstruction set => CollectValueInstanceStrings(set.ValueInstance),
			LoadConstantInstruction loadConst => CollectValueInstanceStrings(loadConst.Constant),
			Invoke { Method: not null } invoke => CollectMethodCallStrings(invoke.Method),
			WriteToListInstruction writeList => Add(writeList.Identifier),
			WriteToTableInstruction writeTable => Add(writeTable.Identifier),
			RemoveInstruction remove => Add(remove.Identifier),
			ListCallInstruction listCall => Add(listCall.Identifier),
			PrintInstruction print => Add(print.TextPrefix),
			_ => this
		};

	public NameTable Add(string name)
	{
		if (indices.TryGetValue(name, out _))
			return this;
		indices[name] = names.Count;
		names.Add(name);
		return this;
	}

	private readonly Dictionary<string, int> indices = new(StringComparer.Ordinal);
	private readonly List<string> names = [];
	public IReadOnlyList<string> Names => names;
	public int this[string name] => indices[name];
	public int Count => names.Count;
	public IEnumerator<string> GetEnumerator() => names.GetEnumerator();
	IEnumerator IEnumerable.GetEnumerator() => GetEnumerator(); //ncrunch: no coverage

	private NameTable CollectValueInstanceStrings(ValueInstance val)
	{
		if (val.IsText)
			return Add(val.Text);
		if (val.IsList)
		{
			Add(val.List.ReturnType.Name);
			foreach (var item in val.List.Items)
				CollectValueInstanceStrings(item);
			return this;
		}
		if (val.IsDictionary)
		{
			Add(val.GetType().Name);
			foreach (var kvp in val.GetDictionaryItems())
			{
				CollectValueInstanceStrings(kvp.Key);
				CollectValueInstanceStrings(kvp.Value);
			}
			return this;
		}
		var type = val.GetType();
		if ((type.IsNone || type.IsBoolean || type.IsNumber || type.IsCharacter) &&
			InstanceInstruction.IsIntegerNumber(val.Number))
			return this;
		return Add(type.Name);
	}

	private NameTable CollectMethodCallStrings(MethodCall mc)
	{
		Add(mc.Method.Type.Name);
		Add(mc.Method.Name);
		Add(mc.ReturnType.Name);
		if (mc.Instance != null)
			CollectExpressionStrings(mc.Instance);
		foreach (var arg in mc.Arguments)
			CollectExpressionStrings(arg);
		return this;
	}

	private NameTable CollectExpressionStrings(Expression? expr) =>
		expr switch
		{
			null => this,
			Value { Data.IsText: true } val => Add(val.Data.Text),
			Value val when val.Data.GetType().IsBoolean => Add(val.Data.GetType().Name),
			Value val when !val.Data.GetType().IsNumber ||
				!InstanceInstruction.IsIntegerNumber(val.Data.Number)
				=> Add(val.Data.GetType().Name), //ncrunch: no coverage
			MemberCall memberCall => Add(memberCall.Member.Name).Add(memberCall.Member.Type.Name).
				CollectExpressionStrings(memberCall.Instance),
			Binary binary => Add(binary.Method.Name). //ncrunch: no coverage
				CollectExpressionStrings(binary.Instance).CollectExpressionStrings(binary.Arguments[0]),
			MethodCall mc => CollectMethodCallStrings(mc),
			_ => Add(expr.ToString()).Add(expr.ReturnType.Name)
		};

	public void Write(BinaryWriter writer)
	{
		writer.Write7BitEncodedInt(Count);
		foreach (var s in this)
			writer.Write(s);
	}
}