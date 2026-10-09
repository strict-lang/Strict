using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	private void ExecuteLoopBegin(LoopBeginInstruction loopBegin)
	{
		if (loopBegin.IsRange)
			ProcessRangeLoopIteration(loopBegin);
		else
			ProcessCollectionLoopIteration(loopBegin);
	}

	private static void CaptureLoopState(LoopBeginInstruction loopBegin, CallFrame frame)
	{
		if (loopBegin.SavedCustomValues != null)
			return;
		loopBegin.SavedIndexValue = frame.TryGet(IndexSymbolId, out var indexValue)
			? indexValue
			: default;
		loopBegin.SavedValue = frame.TryGet(ValueSymbolId, out var value)
			? value
			: default;
		loopBegin.SavedOuterValue = frame.TryGet(OuterSymbolId, out var outerValue)
			? outerValue
			: default;
		loopBegin.SavedOuterIndexValue = frame.TryGet(OuterIndexSymbolId, out var outerIndexValue)
			? outerIndexValue
			: default;
		var savedCustomValues = new Dictionary<string, ValueInstance>(StringComparer.Ordinal);
		for (var variableIndex = 0; variableIndex < loopBegin.CustomVariableNames.Length;
			variableIndex++)
			if (frame.TryGet(loopBegin.CustomVariableNames[variableIndex], out var customValue))
				savedCustomValues.Add(loopBegin.CustomVariableNames[variableIndex], customValue);
		loopBegin.SavedCustomValues = savedCustomValues;
	}

	private static void RestoreLoopState(LoopBeginInstruction loopBegin, CallFrame frame)
	{
		RestoreLoopVariable(frame, IndexSymbolId, Type.IndexLowercase, loopBegin.SavedIndexValue);
		RestoreLoopVariable(frame, ValueSymbolId, Type.ValueLowercase, loopBegin.SavedValue);
		RestoreLoopVariable(frame, OuterSymbolId, Type.OuterLowercase, loopBegin.SavedOuterValue);
		RestoreLoopVariable(frame, OuterIndexSymbolId, Type.OuterLowercase + "." + Type.IndexLowercase,
			loopBegin.SavedOuterIndexValue);
		for (var variableIndex = 0; variableIndex < loopBegin.CustomVariableNames.Length;
			variableIndex++)
		{
			var name = loopBegin.CustomVariableNames[variableIndex];
			RestoreLoopVariable(frame, CallFrame.ResolveSymbolId(name), name,
				loopBegin.SavedCustomValues != null && loopBegin.SavedCustomValues.TryGetValue(name,
					out var customValue)
					? customValue
					: default);
		}
		loopBegin.IsInitialized = false;
		loopBegin.LoopCount = 0;
		loopBegin.ResetIterationState();
	}

	private static void RestoreLoopVariable(CallFrame frame, int symbolId, string name,
		ValueInstance value) =>
		frame.Set(symbolId, value, false, name);

	private void SkipLoopBody()
	{
		var skipTo = instructionIndex + 1;
		while (skipTo < instructions.Count &&
			instructions[skipTo].InstructionType != InstructionType.LoopEnd)
			skipTo++;
		instructionIndex = skipTo;
	}

	private void ProcessCollectionLoopIteration(LoopBeginInstruction loopBegin)
	{
		if (!Memory.Registers.TryGet(loopBegin.Register, out var iterableVariable))
			return;
		var frame = Memory.Frame;
		if (!loopBegin.IsInitialized)
		{
			loopBegin.LoopCount = GetLength(iterableVariable);
			loopBegin.CurrentIndexValue = -1;
			CaptureLoopState(loopBegin, frame);
			loopBegin.IsInitialized = true;
		}
		var nextIndex = (loopBegin.CurrentIndexValue ?? -1) + 1;
		loopBegin.CurrentIndexValue = nextIndex;
		frame.Set(Type.IndexLowercase, new ValueInstance(executable.numberType, nextIndex));
		if (loopBegin.SavedIndexValue.HasValue)
		{
			frame.Set(OuterSymbolId, loopBegin.SavedOuterValue.HasValue
				? loopBegin.SavedOuterValue
				: loopBegin.SavedValue, false, Type.OuterLowercase);
			frame.Set(OuterIndexSymbolId, loopBegin.SavedOuterIndexValue.HasValue
				? loopBegin.SavedOuterIndexValue
				: loopBegin.SavedIndexValue, false, Type.OuterLowercase + "." + Type.IndexLowercase);
		}
		AlterValueVariable(iterableVariable, loopBegin);
		if (loopBegin.LoopCount <= 0)
		{
			RestoreLoopState(loopBegin, frame);
			SkipLoopBody();
		}
		else
			AssignCustomLoopVariables(loopBegin, frame.Get(ValueSymbolId));
	}

	private void ProcessRangeLoopIteration(LoopBeginInstruction loopBegin)
	{
		var frame = Memory.Frame;
		if (!loopBegin.IsInitialized)
		{
			var startIndex = Convert.ToInt32(Memory.Registers[loopBegin.Register].Number);
			var endIndex = Convert.ToInt32(Memory.Registers[loopBegin.EndIndex!.Value].Number);
			loopBegin.InitializeRangeState(startIndex, endIndex);
			CaptureLoopState(loopBegin, frame);
			if (loopBegin.LoopCount <= 0)
			{
				RestoreLoopState(loopBegin, frame);
				SkipLoopBody();
				return;
			}
		}
		var incrementValue = loopBegin.IsDecreasing == true
			? -1
			: 1;
		var currentIndex = loopBegin.CurrentIndexValue.HasValue
			? loopBegin.CurrentIndexValue.Value + incrementValue
			: loopBegin.StartIndexValue ?? 0;
		loopBegin.CurrentIndexValue = currentIndex;
		var currentIndexValue = new ValueInstance(executable.numberType, currentIndex);
		frame.Set(IndexSymbolId, currentIndexValue, false, Type.IndexLowercase);
		frame.Set(ValueSymbolId, currentIndexValue, true, Type.ValueLowercase);
		if (loopBegin.SavedIndexValue.HasValue)
		{
			frame.Set(OuterSymbolId, loopBegin.SavedOuterValue.HasValue
				? loopBegin.SavedOuterValue
				: loopBegin.SavedValue, false, Type.OuterLowercase);
			frame.Set(OuterIndexSymbolId, loopBegin.SavedOuterIndexValue.HasValue
				? loopBegin.SavedOuterIndexValue
				: loopBegin.SavedIndexValue, false, Type.OuterLowercase + "." + Type.IndexLowercase);
		}
		AssignCustomLoopVariables(loopBegin, currentIndexValue);
	}

	private void AssignCustomLoopVariables(LoopBeginInstruction loopBegin, ValueInstance value)
	{
		if (loopBegin.CustomVariableNames.Length == 0)
			return;
		if (loopBegin.CustomVariableNames.Length == 1)
		{
			Memory.Frame.Set(loopBegin.CustomVariableNames[0], value);
			return;
		}
		var loopValues = GetLoopVariableValues(value);
		for (var index = 0; index < loopBegin.CustomVariableNames.Length; index++)
			Memory.Frame.Set(loopBegin.CustomVariableNames[index], loopValues[index]);
	}

	private List<ValueInstance> GetLoopVariableValues(ValueInstance value)
	{
		if (value.IsList)
			return value.List.Items;
		var typeInstance = value.TryGetValueTypeInstance();
		if (typeInstance != null)
			for (var index = 0; index < typeInstance.Values.Length; index++)
				if (!typeInstance.ReturnType.Members[index].IsConstant && typeInstance.Values[index].IsList)
					return typeInstance.Values[index].List.Items;
		throw Fail("Cannot split loop value '" + value +
			"' into variables - expected a list or a type instance with a list member");
	}

	private static int GetLength(ValueInstance iterableInstance) =>
		iterableInstance.GetIteratorLength();

	private void AlterValueVariable(ValueInstance iterableVariable, LoopBeginInstruction loopBegin)
	{
		var frame = Memory.Frame;
		var index = (int)frame.Get(IndexSymbolId).Number;
		if (iterableVariable.IsText)
		{
			if (index < iterableVariable.Text.Length)
				frame.Set(ValueSymbolId, new ValueInstance(iterableVariable.Text[index].ToString()), true,
					Type.ValueLowercase);
			return;
		}
		if (iterableVariable.IsList)
		{
			var list = iterableVariable.List;
			if (index < list.Count)
				frame.Set(ValueSymbolId, list[index], true, Type.ValueLowercase);
			else
				loopBegin.LoopCount = 0;
			return;
		}
		frame.Set(ValueSymbolId, new ValueInstance(executable.numberType, index), true,
			Type.ValueLowercase);
	}

	public sealed class OperandsRequired : Exception;
}
