using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class LoopBeginInstruction : RegisterInstruction
{
	public LoopBeginInstruction(Register register, string customVariableName = "")
		: base(InstructionType.LoopBegin, register) =>
		CustomVariableName = customVariableName;

	public LoopBeginInstruction(Register startIndex, Register endIndex,
		string customVariableName = "")
		: base(InstructionType.LoopBegin, startIndex)
	{
		EndIndex = endIndex;
		CustomVariableName = customVariableName;
	}

	public LoopBeginInstruction(BinaryReader reader, NameTable table)
		: this((Register)reader.ReadByte())
	{
		if (reader.ReadBoolean())
			EndIndex = (Register)reader.Read7BitEncodedInt();
		CustomVariableName = table.names[reader.Read7BitEncodedInt()];
	}

	public Register? EndIndex { get; }
	public string CustomVariableName { get; }
	public bool IsRange => EndIndex != null;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write(EndIndex != null);
		if (EndIndex != null)
			writer.Write7BitEncodedInt((int)EndIndex!.Value);
		writer.Write7BitEncodedInt(table[CustomVariableName]);
	}

	public bool IsInitialized { get; set; }
	public int LoopCount { get; set; }
	public int InstructionIndex { get; set; } = -1;
	public int? StartIndexValue { get; private set; }
	public int? EndIndexValue { get; private set; }
	public bool? IsDecreasing { get; private set; }

	public void InitializeRangeState(int startIndex, int endIndex)
	{
		StartIndexValue = startIndex;
		EndIndexValue = endIndex;
		IsDecreasing = endIndex < startIndex;
		LoopCount = IsDecreasing.Value
			? startIndex - endIndex
			: endIndex - startIndex;
		IsInitialized = true;
	}

	public void Reset()
	{
		IsInitialized = false;
		LoopCount = 0;
		InstructionIndex = -1;
		StartIndexValue = null;
		EndIndexValue = null;
		IsDecreasing = null;
	}
}