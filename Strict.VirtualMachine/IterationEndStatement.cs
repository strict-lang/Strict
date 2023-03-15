namespace Strict.VirtualMachine;

public sealed class IterationEndStatement : RegisterStatement
{
	public IterationEndStatement(Register register) : base(register, Instruction.IterationEnd) => IteratorRegister = register;
	public Register IteratorRegister { get; } //ncrunch: no coverage, TODO: missing tests
}