using System.Runtime.CompilerServices;
using Strict.Bytecode.Instructions;

[assembly: InternalsVisibleTo("Strict.Optimizers")]

namespace Strict.Bytecode.Serialization;

public record BinaryMethod
{
	public BinaryMethod(BinaryReader reader, BinaryType type, string methodName)
	{
		type.ReadMembers(reader, parameters);
		ReturnTypeName = type.Table.Names[reader.Read7BitEncodedInt()];
		//TODO: remove: EnsureMethod(type, methodName, parameters.Select(parameter => parameter.Name + " " + parameter.FullTypeName).ToArray(), returnTypeName);
		var instructionCount = reader.Read7BitEncodedInt();
		for (var instructionIndex = 0; instructionIndex < instructionCount; instructionIndex++)
			instructions.Add(type.binary.ReadInstruction(reader, type.Table));
	}

	private readonly List<BinaryMember> parameters = [];
	public IReadOnlyList<BinaryMember> Parameters => parameters;
	public string ReturnTypeName { get; }
	internal List<Instruction> instructions = [];
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