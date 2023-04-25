namespace Strict.VirtualMachine;

public sealed class LoopBeginStatementRange : Statement
{
	public LoopBeginStatementRange(Register startIndex, Register endIndex)
	{
		StartIndex = startIndex;
		EndIndex = endIndex;
	}

	public Register StartIndex { get; set; }
	public Register EndIndex { get; set; }
	public override Instruction Instruction => Instruction.LoopBeginRange;
}