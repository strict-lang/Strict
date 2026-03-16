using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class BinaryInstruction(InstructionType instructionType, params Register[] registers)
	: Instruction(instructionType)
{
	public BinaryInstruction(BinaryReader reader, InstructionType instructionType)
		: this(instructionType, ReadRegisters(reader)) { }

	private static Register[] ReadRegisters(BinaryReader reader)
	{
		var registersCount = reader.ReadByte();
		var registers = new Register[registersCount];
		for (var index = 0; index < registersCount; index++)
			registers[index] = (Register)reader.ReadByte();
		return registers;
	}

	public Register[] Registers { get; } = registers;
	public override string ToString() => $"{InstructionType} {string.Join(" ", Registers)}";

	public bool IsConditional() =>
		InstructionType is > InstructionType.ArithmeticSeparator
			and < InstructionType.BinaryOperatorsSeparator;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write((byte)Registers.Length);
		foreach (var register in Registers)
			writer.Write((byte)register);
	}
}