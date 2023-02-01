namespace Strict.VirtualMachine;

public sealed class JumpToIdStatement : JumpStatement
{
	public JumpToIdStatement(Instruction jumpInstruction, int id) : base(jumpInstruction) =>
		Id = id;

	public int Id { get; }
}