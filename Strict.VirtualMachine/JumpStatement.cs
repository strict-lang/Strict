namespace Strict.VirtualMachine;

public class JumpStatement : Statement
{
	protected JumpStatement(Instruction jumpInstruction) => Instruction = jumpInstruction;
	public override Instruction Instruction { get; }
}