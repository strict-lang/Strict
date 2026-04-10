using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// Used in BytecodeSerializer to write out strings, names, identifiers, etc. once.
/// </summary>
public sealed class NameTable
{
	public NameTable(BinaryReader reader, string justTypeName) : this(justTypeName)
	{
		var customNamesCount = reader.Read7BitEncodedInt();
		for (var index = 0; index < customNamesCount; index++)
			Add(reader.ReadString());
	}

	public NameTable(string justTypeName)
	{
		foreach (var predefinedName in BuiltInPredefinedNames)
			Add(predefinedName);
		Add(justTypeName);
		prefilledNamesCount = names.Count;
	}

	private readonly int prefilledNamesCount;
	/// <summary>
	/// Common names that appear in most .strict files, mostly base type usages.
	/// </summary>
	private static readonly string[] BuiltInPredefinedNames =
	[
		"",
		Type.None,
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
		Type.File,
		Type.Directory,
		Type.TextWriter,
		Type.TextReader,
		Type.Stacktrace,
		Type.Mutable,
		Type.Dictionary,
		nameof(Strict) + Context.ParentSeparator + Type.None,
		nameof(Strict) + Context.ParentSeparator + Type.Boolean,
		nameof(Strict) + Context.ParentSeparator + Type.Number,
		nameof(Strict) + Context.ParentSeparator + Type.Character,
		nameof(Strict) + Context.ParentSeparator + Type.Range,
		nameof(Strict) + Context.ParentSeparator + Type.Text,
		nameof(Strict) + Context.ParentSeparator + Type.Error,
		nameof(Strict) + Context.ParentSeparator + Type.ErrorWithValue,
		nameof(Strict) + Context.ParentSeparator + Type.Iterator,
		nameof(Strict) + Context.ParentSeparator + Type.List,
		nameof(Strict) + Context.ParentSeparator + Type.Logger,
		nameof(Strict) + Context.ParentSeparator + Type.File,
		nameof(Strict) + Context.ParentSeparator + Type.Directory,
		nameof(Strict) + Context.ParentSeparator + Type.TextWriter,
		nameof(Strict) + Context.ParentSeparator + Type.TextReader,
		nameof(Strict) + Context.ParentSeparator + Type.Stacktrace,
		nameof(Strict) + Context.ParentSeparator + Type.Mutable,
		nameof(Strict) + Context.ParentSeparator + Type.Dictionary,
		Type.Boolean.MakeFirstLetterLowercase(),
		Type.Number.MakeFirstLetterLowercase(),
		Type.Character.MakeFirstLetterLowercase(),
		Type.Range.MakeFirstLetterLowercase(),
		Type.Text.MakeFirstLetterLowercase(),
		Type.Error.MakeFirstLetterLowercase(),
		Type.Iterator.MakeFirstLetterLowercase(),
		Type.List.MakeFirstLetterLowercase(),
		Type.Logger.MakeFirstLetterLowercase(),
		Type.File.MakeFirstLetterLowercase(),
		Type.Directory.MakeFirstLetterLowercase(),
		Type.TextWriter.MakeFirstLetterLowercase(),
		Type.TextReader.MakeFirstLetterLowercase(),
		Type.Stacktrace.MakeFirstLetterLowercase(),
		Type.Mutable.MakeFirstLetterLowercase(),
		Type.Dictionary.MakeFirstLetterLowercase(),
		Type.IndexLowercase,
		Type.ValueLowercase,
		Method.Run,
		Method.From,
		"first",
		"second",
		"numbers",
		"characters",
		"texts",
		"NewLine",
		"Tab",
		nameof(Strict) + Context.ParentSeparator + Type.List + "(" + Type.Number + ")",
		Type.List + "(" + Type.Number + ")",
		nameof(Strict) + Context.ParentSeparator + Type.List + "(" + Type.Text + ")",
		Type.List + "(" + Type.Text + ")"
	];

	public NameTable CollectStrings(Instruction instruction) =>
		instruction switch
		{
			LoadVariableToRegister loadVar => Add(loadVar.Identifier),
			StoreVariableInstruction storeVar => Add(storeVar.Identifier).
				CollectValueInstanceStrings(storeVar.ValueInstance),
			StoreFromRegisterInstruction storeReg => Add(storeReg.Identifier),
			SetInstruction set => CollectValueInstanceStrings(set.ValueInstance),
			LoadConstantInstruction loadConst => CollectValueInstanceStrings(loadConst.Constant),
			Invoke invoke => CollectInvokeMethodInfoStrings(invoke.MethodInfo),
			WriteToListInstruction writeList => Add(writeList.Identifier),
			WriteToTableInstruction writeTable => Add(writeTable.Identifier),
			RemoveInstruction remove => Add(remove.Identifier),
			ListCallInstruction listCall => Add(listCall.Identifier),
			PrintInstruction print => Add(print.TextPrefix),
			LoopBeginInstruction loopBegin => loopBegin.CustomVariableNames.Aggregate(this,
				(current, customVariableName) => current.Add(customVariableName)),
			_ => this
		};

	public NameTable Add(string name)
	{
		if (indices.ContainsKey(name))
			return this;
		indices[name] = names.Count;
		names.Add(name);
		return this;
	}

	private readonly Dictionary<string, int> indices = new(StringComparer.Ordinal);
	public readonly List<string> names = [];
	public int this[string name] => indices[name];

	private NameTable CollectValueInstanceStrings(ValueInstance val)
	{
		if (val.IsText)
			return Add(val.Text);
		if (val.IsList)
		{
			Add(val.List.ReturnType.FullName);
			foreach (var item in val.List.Items)
				CollectValueInstanceStrings(item);
			return this;
		}
		if (val.IsDictionary)
		{
			Add(val.GetType().FullName);
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
		return Add(type.FullName);
	}

	private NameTable CollectInvokeMethodInfoStrings(InvokeMethodInfo info)
	{
		Add(info.TypeFullName);
		Add(info.MethodName);
		Add(info.ReturnTypeName);
		foreach (var paramName in info.ParameterNames)
			Add(paramName);
		return this;
	}

	private NameTable CollectMethodCallStrings(MethodCall mc)
	{
		Add(mc.Method.Type.FullName);
		Add(mc.Method.Name);
		Add(mc.ReturnType.FullName);
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
			List list => Add(list.ReturnType.FullName).CollectListExpressionStrings(list),
			Value { Data.IsText: true } val => Add(val.Data.Text),
			Value val when val.Data.GetType().IsBoolean => Add(val.Data.GetType().Name),
			Value val when !val.Data.GetType().IsNumber ||
				!BinaryExecutable.IsIntegerNumber(val.Data.Number) => Add(val.Data.GetType().Name),
			//TODO: need tests!
			MemberCall memberCall => Add(memberCall.Member.Name).Add(memberCall.Member.Type.FullName).
				CollectExpressionStrings(memberCall.Instance),
			Binary binary => Add(binary.Method.Name).CollectExpressionStrings(binary.Instance).
				CollectExpressionStrings(binary.Arguments[0]),
			MethodCall mc => CollectMethodCallStrings(mc),
			ListCall listCall => Add(listCall.ReturnType.FullName).CollectExpressionStrings(listCall.List).
				CollectExpressionStrings(listCall.Index),
			_ => Add(expr.ToString()).Add(expr.ReturnType.FullName)
		};

	//TODO: never called, even needed?
	private NameTable CollectListExpressionStrings(List list)
	{
		foreach (var value in list.Values)
			CollectExpressionStrings(value);
		return this;
	}

	public void Write(BinaryWriter writer)
	{
		var customNamesCount = names.Count - prefilledNamesCount;
		writer.Write7BitEncodedInt(customNamesCount);
		for (var index = prefilledNamesCount; index < names.Count; index++)
			writer.Write(names[index]);
	}
}