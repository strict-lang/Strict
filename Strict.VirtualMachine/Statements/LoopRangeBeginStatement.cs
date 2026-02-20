namespace Strict.Runtime.Statements;

public sealed class LoopRangeBeginStatement(Register startIndex, Register endIndex)
	: Statement(Instruction.LoopBeginRange)
{
	public Register StartIndex { get; } = startIndex;
	public Register EndIndex { get; } = endIndex;
}
