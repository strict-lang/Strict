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
	public int? StartIndexValue { get; private set; }
	public int? EndIndexValue { get; private set; }
	public bool? IsDecreasing { get; private set; }

	public void InitializeRangeState(int startIndex, int endIndex)
	{
		StartIndexValue = startIndex;
		EndIndexValue = endIndex;
		IsDecreasing = endIndex < startIndex;
		LoopCount = (IsDecreasing.Value
			? startIndex - endIndex
			: endIndex - startIndex) + 1;
		IsInitialized = true;
	}

	public void Reset()
	{
		IsInitialized = false;
		LoopCount = 0;
		StartIndexValue = null;
		EndIndexValue = null;
		IsDecreasing = null;
	}
}
