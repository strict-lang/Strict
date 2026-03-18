using System.Collections;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// Used in BytecodeSerializer to write out strings, names, identifiers, etc. once.
/// </summary>
public sealed class NameTable : IEnumerable<string>
{
	public NameTable(IEnumerable<string>? predefinedNames = null)
	{
		AddBuiltInPredefinedNames();
		if (predefinedNames != null)
			AddPredefinedNames(predefinedNames);
	}

	public NameTable(BinaryReader reader, IEnumerable<string>? predefinedNames = null) :
		this(predefinedNames)
	{
		var customNamesCount = reader.Read7BitEncodedInt();
		for (var index = 0; index < customNamesCount; index++)
			Add(reader.ReadString());
	}

	private void AddPredefinedNames(IEnumerable<string> predefinedNames)
	{
		foreach (var predefinedName in predefinedNames)
			Add(predefinedName);
		predefinedNamesCount = names.Count;
	}

	private static readonly string[] BuiltInPredefinedNames =
	[
		"",
		Type.None,
		Type.Any,
		Type.Boolean,
		Type.Number,
		Type.Character,
		Type.Range,
		Type.Text,
		Type.Error,
		Type.ErrorWithValue,
		Type.Iterator,
		Type.List,
		Type.Logger,
		Type.App,
		Type.System,
		Type.File,
		Type.Directory,
		Type.TextWriter,
		Type.TextReader,
		Type.Stacktrace,
		Type.Mutable,
		Type.Dictionary,
		Type.None.ToLowerInvariant(),
		Type.Any.ToLowerInvariant(),
		Type.Boolean.ToLowerInvariant(),
		Type.Number.ToLowerInvariant(),
		Type.Character.ToLowerInvariant(),
		Type.Range.ToLowerInvariant(),
		Type.Text.ToLowerInvariant(),
		Type.Error.ToLowerInvariant(),
		Type.ErrorWithValue.ToLowerInvariant(),
		Type.Iterator.ToLowerInvariant(),
		Type.List.ToLowerInvariant(),
		Type.Logger.ToLowerInvariant(),
		Type.App.ToLowerInvariant(),
		Type.System.ToLowerInvariant(),
		Type.File.ToLowerInvariant(),
		Type.Directory.ToLowerInvariant(),
		Type.TextWriter.ToLowerInvariant(),
		Type.TextReader.ToLowerInvariant(),
		Type.Stacktrace.ToLowerInvariant(),
		Type.Mutable.ToLowerInvariant(),
		Type.Dictionary.ToLowerInvariant(),
		"first",
		"second",
		"from",
		"Run",
		"characters",
		"Strict/List(Character)",
		"Strict/List(Number)",
		"Strict/List(Text)",
		"zeroCharacter",
		"NewLine",
		"Tab",
		"textWriter"
	];

	private void AddBuiltInPredefinedNames()
	{
		foreach (var predefinedName in BuiltInPredefinedNames)
			Add(predefinedName);
		predefinedNamesCount = names.Count;
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
			Invoke invoke => CollectMethodCallStrings(invoke.Method),
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
	private int predefinedNamesCount;
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
			BinaryExecutable.IsIntegerNumber(val.Number))
			return this;
		return Add(type.Name);
	}

	private NameTable CollectMethodCallStrings(MethodCall mc)
	{
		Add(mc.Method.Type.Name);
		Add(mc.Method.Name);
		Add(mc.ReturnType.Name);
    foreach (var parameter in mc.Method.Parameters)
			Add(parameter.Name).Add(parameter.Type.FullName);
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
			Value val when !val.Data.GetType().IsNumber || !BinaryExecutable.IsIntegerNumber(val.Data.Number)
				=> Add(val.Data.GetType().Name), //ncrunch: no coverage
			MemberCall memberCall => Add(memberCall.Member.Name).Add(memberCall.Member.Type.Name).
				CollectExpressionStrings(memberCall.Instance),
			Expressions.Binary binary => Add(binary.Method.Name). //ncrunch: no coverage
				CollectExpressionStrings(binary.Instance).CollectExpressionStrings(binary.Arguments[0]),
			MethodCall mc => CollectMethodCallStrings(mc),
			_ => Add(expr.ToString()).Add(expr.ReturnType.Name)
		};

	public void Write(BinaryWriter writer)
	{
		var customNamesCount = names.Count - predefinedNamesCount;
		writer.Write7BitEncodedInt(customNamesCount);
		for (var index = predefinedNamesCount; index < names.Count; index++)
			writer.Write(names[index]);
	}
}