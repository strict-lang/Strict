using Strict.Bytecode.Instructions;

namespace Strict.Bytecode.Serialization;

public sealed class BinaryMethod
{
	public BinaryMethod(string methodName, List<BinaryMember> methodParameters,
		string returnTypeName, List<Instruction> methodInstructions)
	{
		Name = methodName;
		ReturnTypeName = returnTypeName;
		parameters = methodParameters;
		instructions = methodInstructions;
	}

	public string Name { get; }
	public string ReturnTypeName { get; }
	public readonly List<BinaryMember> parameters = [];
	public List<Instruction> instructions = [];

	public BinaryMethod(BinaryReader reader, BinaryType type, string methodName)
	{
		Name = methodName;
		type.ReadMembers(reader, parameters);
		ReturnTypeName = type.Table.names[reader.Read7BitEncodedInt()];
		var instructionCount = reader.Read7BitEncodedInt();
		for (var instructionIndex = 0; instructionIndex < instructionCount; instructionIndex++)
			instructions.Add(type.binary!.ReadInstruction(reader, type.Table));
   RestoreLoopLinks();
	}

	private void RestoreLoopLinks()
	{
		var loopBeginIndexes = new Stack<int>();
		for (var instructionIndex = 0; instructionIndex < instructions.Count; instructionIndex++)
			switch (instructions[instructionIndex])
			{
			case LoopBeginInstruction loopBegin:
				loopBegin.InstructionIndex = instructionIndex;
				loopBeginIndexes.Push(instructionIndex);
				break;
			case LoopEndInstruction loopEnd when loopBeginIndexes.Count > 0:
				var beginIndex = loopBeginIndexes.Pop();
				loopEnd.Begin = (LoopBeginInstruction)instructions[beginIndex];
				loopEnd.BeginIndex = beginIndex;
				break;
			}
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