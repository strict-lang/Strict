namespace Strict.VirtualMachine;

public sealed class LoopRangeBeginStatement(Register startIndex, Register endIndex) : Statement
{
	public Register StartIndex { get; set; } = startIndex;
	public Register EndIndex { get; set; } = endIndex;
	public override Instruction Instruction => Instruction.LoopBeginRange;
}