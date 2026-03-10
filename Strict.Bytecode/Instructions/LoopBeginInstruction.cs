namespace Strict.Bytecode.Instructions;

public sealed class LoopBeginInstruction : RegisterInstruction
{
	public LoopBeginInstruction(Register register) : base(InstructionType.LoopBegin, register) { }

	public LoopBeginInstruction(Register startIndex, Register endIndex)
		: base(InstructionType.LoopBegin, startIndex)
	{
		EndIndex = endIndex;
		IsRange = true;
	}

	public Register? EndIndex { get; }
	public bool IsRange { get; }
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