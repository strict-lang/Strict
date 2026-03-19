using Strict.Bytecode.Instructions;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

namespace Strict.Bytecode.Serialization;

public record BinaryMethod
{
	public BinaryMethod(string methodName, List<BinaryMember> methodParameters,
		string returnTypeName, List<Instruction> methodInstructions)
	{
		//TODO: methodName should not be null or empty, Run is the default, not ""
		Name = methodName;
		ReturnTypeName = returnTypeName;
		parameters = methodParameters;
		instructions = methodInstructions;
	}

	public string Name { get; }
	public string ReturnTypeName { get; }
	public List<BinaryMember> parameters = [];
	public List<Instruction> instructions = [];

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