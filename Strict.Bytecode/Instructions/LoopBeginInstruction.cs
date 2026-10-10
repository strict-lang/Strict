using Strict.Bytecode.Serialization;
using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

public sealed class LoopBeginInstruction : RegisterInstruction
{
	public LoopBeginInstruction(Register register, params string[] customVariableNames) :
		base(InstructionType.LoopBegin, register) =>
		CustomVariableNames = customVariableNames;

	public LoopBeginInstruction(Register startIndex, Register endIndex,
		params string[] customVariableNames) : base(InstructionType.LoopBegin, startIndex)
	{
		EndIndex = endIndex;
		CustomVariableNames = customVariableNames;
	}

	public LoopBeginInstruction(BinaryReader reader, NameTable table) : this(
		(Register)reader.ReadByte())
	{
		if (reader.ReadBoolean())
			EndIndex = (Register)reader.Read7BitEncodedInt();
		var customVariableCount = reader.Read7BitEncodedInt();
		CustomVariableNames = new string[customVariableCount];
		for (var index = 0; index < customVariableCount; index++)
			CustomVariableNames[index] = table.names[reader.Read7BitEncodedInt()];
	}

	public Register? EndIndex { get; }
	public string[] CustomVariableNames { get; }
	public bool IsRange => EndIndex != null;

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write(EndIndex != null);
		if (EndIndex != null)
			writer.Write7BitEncodedInt((int)EndIndex!.Value);
		writer.Write7BitEncodedInt(CustomVariableNames.Length);
		for (var index = 0; index < CustomVariableNames.Length; index++)
			writer.Write7BitEncodedInt(table[CustomVariableNames[index]]);
	}

	public bool IsInitialized { get; set; }
	public int LoopCount { get; set; }
	public int InstructionIndex { get; set; } = -1;
	public int? StartIndexValue { get; private set; }
	public int? EndIndexValue { get; private set; }
	public bool? IsDecreasing { get; private set; }
	public int? CurrentIndexValue { get; set; }
	public ValueInstance SavedIndexValue { get; set; }
	public ValueInstance SavedValue { get; set; }
	public ValueInstance SavedOuterValue { get; set; }
	public ValueInstance SavedOuterIndexValue { get; set; }
	public Dictionary<string, ValueInstance>? SavedCustomValues { get; set; }

	public void InitializeRangeState(int startIndex, int endIndex)
	{
		StartIndexValue = startIndex;
		EndIndexValue = endIndex;
		IsDecreasing = endIndex < startIndex;
		LoopCount = IsDecreasing.Value
			? startIndex - endIndex
			: endIndex - startIndex;
		CurrentIndexValue = null;
		IsInitialized = true;
	}

	/// <summary>
	/// ponytail: runtime loop state lives on the shared instruction, a re-entrant call of the same
	/// method saves and restores it. Upgrade path: keep the loop state per call frame in the VM.
	/// </summary>
	public State SaveState() =>
		new(IsInitialized, LoopCount, InstructionIndex, StartIndexValue, EndIndexValue, IsDecreasing,
			CurrentIndexValue, SavedIndexValue, SavedValue, SavedOuterValue, SavedOuterIndexValue,
			SavedCustomValues);

	public void RestoreState(State state)
	{
		IsInitialized = state.IsInitialized;
		LoopCount = state.LoopCount;
		InstructionIndex = state.InstructionIndex;
		StartIndexValue = state.StartIndexValue;
		EndIndexValue = state.EndIndexValue;
		IsDecreasing = state.IsDecreasing;
		CurrentIndexValue = state.CurrentIndexValue;
		SavedIndexValue = state.SavedIndexValue;
		SavedValue = state.SavedValue;
		SavedOuterValue = state.SavedOuterValue;
		SavedOuterIndexValue = state.SavedOuterIndexValue;
		SavedCustomValues = state.SavedCustomValues;
	}

	public readonly record struct State(bool IsInitialized, int LoopCount, int InstructionIndex,
		int? StartIndexValue, int? EndIndexValue, bool? IsDecreasing, int? CurrentIndexValue,
		ValueInstance SavedIndexValue, ValueInstance SavedValue, ValueInstance SavedOuterValue,
		ValueInstance SavedOuterIndexValue, Dictionary<string, ValueInstance>? SavedCustomValues);

	public void Reset()
	{
		IsInitialized = false;
		LoopCount = 0;
		InstructionIndex = -1;
		ResetIterationState();
	}

	public void ResetIterationState()
	{
		StartIndexValue = null;
		EndIndexValue = null;
		IsDecreasing = null;
		CurrentIndexValue = null;
		SavedIndexValue = default;
		SavedValue = default;
		SavedOuterValue = default;
		SavedOuterIndexValue = default;
		SavedCustomValues = null;
	}
}