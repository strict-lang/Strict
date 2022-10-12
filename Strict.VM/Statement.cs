namespace Strict.VM;

//TODO: should be a record
public sealed class Statement
{
	public Statement(Instruction instruction, double value = 0) : this(instruction, Register.None,
		value) { }

	public Statement(Instruction instruction, Register register, double value = 0)
	{
		Instruction = instruction;
		Register = register;
		Value = value;
	}

	public Instruction Instruction { get; }
	public Register Register { get; }
	public double Value { get; }
}