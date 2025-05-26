namespace Strict.VirtualMachine;

public abstract class RegisterStatement(Register register, Instruction instruction) : Statement
{
	public Register Register { get; } = register;
	public override Instruction Instruction { get; } = instruction;
	public override string ToString() => $"{Instruction} {Register}";
}