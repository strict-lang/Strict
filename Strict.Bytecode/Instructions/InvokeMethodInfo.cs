using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// Pure data structure that stores everything the VirtualMachine needs to execute a method call,
/// without any reference to Strict.Expressions types. Replaces the MethodCall expression that was
/// previously embedded in Invoke instructions, eliminating shared expression state that caused
/// thread-safety issues in parallel test execution.
/// </summary>
public sealed class InvokeMethodInfo
{
	public InvokeMethodInfo(string typeFullName, string methodName, string[] parameterNames,
		string returnTypeName, Register[] argumentRegisters, Register? instanceRegister)
	{
		TypeFullName = typeFullName;
		MethodName = methodName;
		ParameterNames = parameterNames;
		ReturnTypeName = returnTypeName;
		ArgumentRegisters = argumentRegisters;
		InstanceRegister = instanceRegister;
	}

	public InvokeMethodInfo(BinaryReader reader, NameTable table)
	{
		TypeFullName = table.names[reader.Read7BitEncodedInt()];
		MethodName = table.names[reader.Read7BitEncodedInt()];
		var paramCount = reader.Read7BitEncodedInt();
		ParameterNames = new string[paramCount];
		for (var index = 0; index < paramCount; index++)
			ParameterNames[index] = table.names[reader.Read7BitEncodedInt()];
		ReturnTypeName = table.names[reader.Read7BitEncodedInt()];
		var argCount = reader.Read7BitEncodedInt();
		ArgumentRegisters = new Register[argCount];
		for (var index = 0; index < argCount; index++)
			ArgumentRegisters[index] = (Register)reader.ReadByte();
		InstanceRegister = reader.ReadBoolean()
			? (Register)reader.ReadByte()
			: null;
	}

	public string TypeFullName { get; }
	public string MethodName { get; }
	public string[] ParameterNames { get; }
	public string ReturnTypeName { get; }
	public Register[] ArgumentRegisters { get; }
	public Register? InstanceRegister { get; }
	internal Type? ResolvedReturnType { get; set; }

	public void Write(BinaryWriter writer, NameTable table)
	{
		writer.Write7BitEncodedInt(table[TypeFullName]);
		writer.Write7BitEncodedInt(table[MethodName]);
		writer.Write7BitEncodedInt(ParameterNames.Length);
		foreach (var paramName in ParameterNames)
			writer.Write7BitEncodedInt(table[paramName]);
		writer.Write7BitEncodedInt(table[ReturnTypeName]);
		writer.Write7BitEncodedInt(ArgumentRegisters.Length);
		foreach (var argRegister in ArgumentRegisters)
			writer.Write((byte)argRegister);
		writer.Write(InstanceRegister.HasValue);
		if (InstanceRegister.HasValue)
			writer.Write((byte)InstanceRegister.Value);
	}

	public Type ResolveReturnType(Package basePackage) =>
		ResolvedReturnType ??= basePackage.FindType(ReturnTypeName) ??
			basePackage.FindFullType(ReturnTypeName) ??
			basePackage.FindType(GetSimpleTypeName(ReturnTypeName)) ??
			basePackage.GetType(Type.None);

	public Type ResolveDeclaringType(Package basePackage) =>
		basePackage.FindFullType(TypeFullName) ?? basePackage.FindType(TypeFullName) ??
			basePackage.FindType(GetSimpleTypeName(TypeFullName)) ??
			basePackage.GetType(Type.None);

	private static string GetSimpleTypeName(string fullTypeName)
	{
		var separatorIndex = fullTypeName.LastIndexOf(Context.ParentSeparator);
		return separatorIndex >= 0
			? fullTypeName[(separatorIndex + 1)..]
			: fullTypeName;
	}

	public override string ToString()
	{
		if (MethodName == Method.From)
		{
			var args = ArgumentRegisters.Length > 0
				? string.Join(", ", ArgumentRegisters.Select(register => register.ToString()))
				: "";
			var simpleTypeName = TypeFullName.Contains(Context.ParentSeparator)
				? TypeFullName[(TypeFullName.LastIndexOf(Context.ParentSeparator) + 1)..]
				: TypeFullName;
			return simpleTypeName + "(" + args + ")";
		}
		if (InstanceRegister.HasValue)
			return InstanceRegister.Value + "." + MethodName;
		return TypeFullName + "." + MethodName + "(" + ArgumentRegisters.Length + " args)";
	}}
