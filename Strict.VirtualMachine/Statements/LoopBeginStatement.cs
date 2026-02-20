namespace Strict.Runtime.Statements;

public sealed class LoopBeginStatement : RegisterStatement
{
	public LoopBeginStatement(Register register) : base(Instruction.LoopBegin, register) { }

	/// <summary>Range loop: from startIndex to endIndex register (inclusive).</summary>
	public LoopBeginStatement(Register startIndex, Register endIndex)
		: base(Instruction.LoopBegin, startIndex)
	{
		EndIndex = endIndex;
		IsRange = true;
	}

	public Register? EndIndex { get; }
	public bool IsRange { get; }
	/// <summary>Loop execution state - set during Execute, reset before each new run.</summary>
	public bool IsInitialized { get; set; }
	public int LoopCount { get; set; }

	public void Reset()
	{
		IsInitialized = false;
		LoopCount = 0;
	}
}
