using Strict.Bytecode.Instructions;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Optimizers")]
[assembly: InternalsVisibleTo("Strict.Compiler")]

namespace Strict.Bytecode.Serialization;

public record BinaryMethod
{
	public BinaryMethod(string methodName, List<BinaryMember> methodParameters,
		string returnTypeName, List<Instruction> methodInstructions)
	{
		Name = methodName;
		ReturnTypeName = returnTypeName;
		parameters = methodParameters;
		instructions = methodInstructions;
	}

	public BinaryMethod(string methodName, IReadOnlyList<BinaryMember> methodParameters,
		string returnTypeName, IReadOnlyList<Instruction> methodInstructions)
	{
		Name = methodName;
		ReturnTypeName = returnTypeName;
		parameters = [.. methodParameters];
		instructions = [.. methodInstructions];
	}

	public string Name { get; }
	public string ReturnTypeName { get; }
	internal List<BinaryMember> parameters = [];
	internal List<Instruction> instructions = [];
	public IReadOnlyList<BinaryMember> Parameters => parameters;
	public IReadOnlyList<Instruction> Instructions => instructions;

	public BinaryMethod(BinaryReader reader, BinaryType type, string methodName)
	{
		Name = methodName;
		type.ReadMembers(reader, parameters);
		ReturnTypeName = type.Table.Names[reader.Read7BitEncodedInt()];
		var instructionCount = reader.Read7BitEncodedInt();
		for (var instructionIndex = 0; instructionIndex < instructionCount; instructionIndex++)
			instructions.Add(type.binary!.ReadInstruction(reader, type.Table));
	}

	public bool UsesConsolePrint => instructions.Any(instruction => instruction is PrintInstruction);

	public void Write(BinaryWriter writer, BinaryType type)
	{
		type.WriteMembers(writer, parameters);
		writer.Write7BitEncodedInt(type.Table[ReturnTypeName]);
		writer.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			instruction.Write(writer, type.Table);
	}
}