namespace Strict.VirtualMachine;

public abstract class RegisterStatement : Statement
{
	protected RegisterStatement(Register register, Instruction instruction)
	{
		Register = register;
		Instruction = instruction;
	}

	public Register Register { get; }
	public override Instruction Instruction { get; }
	public override string ToString() => $"{Instruction} {Register}";
}