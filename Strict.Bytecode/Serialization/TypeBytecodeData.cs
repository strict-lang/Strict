using Strict.Bytecode.Instructions;

namespace Strict.Bytecode.Serialization;

public sealed class TypeBytecodeData(string typeName, string entryPath,
	IReadOnlyList<MemberBytecodeData> members, IReadOnlyList<MethodBytecodeData> methods,
	IList<Instruction> runInstructions,
	IReadOnlyDictionary<MethodBytecodeData, IList<Instruction>> methodInstructions)
{
	public string TypeName { get; } = typeName;
	public string EntryPath { get; } = entryPath;
	public IReadOnlyList<MemberBytecodeData> Members { get; } = members;
	public IReadOnlyList<MethodBytecodeData> Methods { get; } = methods;
	public IList<Instruction> RunInstructions { get; } = runInstructions;
	public IReadOnlyDictionary<MethodBytecodeData, IList<Instruction>> MethodInstructions { get; } =
		methodInstructions;
}

public sealed class MemberBytecodeData(string name, string typeName)
{
	public string Name { get; } = name;
	public string TypeName { get; } = typeName;
}

public sealed class MethodBytecodeData(string name,
	IReadOnlyList<MethodParameterBytecodeData> parameters, string returnTypeName)
{
	public string Name { get; } = name;
	public IReadOnlyList<MethodParameterBytecodeData> Parameters { get; } = parameters;
	public string ReturnTypeName { get; } = returnTypeName;
}

public sealed class MethodParameterBytecodeData(string name, string typeName)
{
	public string Name { get; } = name;
	public string TypeName { get; } = typeName;
}
