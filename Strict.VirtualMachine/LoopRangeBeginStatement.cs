namespace Strict.VirtualMachine;

public sealed class LoopRangeBeginStatement : Statement
{
	public LoopRangeBeginStatement(Register startIndex, Register endIndex)
	{
		StartIndex = startIndex;
		EndIndex = endIndex;
	}

	public Register StartIndex { get; set; }
	public Register EndIndex { get; set; }
	public override Instruction Instruction => Instruction.LoopBeginRange;
}