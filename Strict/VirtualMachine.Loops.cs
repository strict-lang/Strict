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
		RestoreLoopVariable(frame, OuterIndexSymbolId,
			Type.OuterLowercase + "." + Type.IndexLowercase, loopBegin.SavedOuterIndexValue);
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
		if (loopBegin.SavedValue.HasValue)
			frame.Set(OuterSymbolId, loopBegin.SavedValue, false, Type.OuterLowercase);
		if (loopBegin.SavedIndexValue.HasValue)
			frame.Set(OuterIndexSymbolId, loopBegin.SavedIndexValue, false,
				Type.OuterLowercase + "." + Type.IndexLowercase);
		AlterValueVariable(iterableVariable, loopBegin);
		AssignCustomLoopVariables(loopBegin, frame.Get(ValueSymbolId));
		if (loopBegin.LoopCount <= 0)
		{
			RestoreLoopState(loopBegin, frame);
			SkipLoopBody();
		}
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
		if (loopBegin.SavedValue.HasValue)
			frame.Set(OuterSymbolId, loopBegin.SavedValue, false, Type.OuterLowercase);
		if (loopBegin.SavedIndexValue.HasValue)
			frame.Set(OuterIndexSymbolId, loopBegin.SavedIndexValue, false,
				Type.OuterLowercase + "." + Type.IndexLowercase);
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

	private static IReadOnlyList<ValueInstance> GetLoopVariableValues(ValueInstance value)
	{
		if (value.IsList)
			return value.List.Items;
		var typeInstance = value.TryGetValueTypeInstance();
		if (typeInstance != null)
			for (var index = 0; index < typeInstance.Values.Length; index++)
				if (!typeInstance.ReturnType.Members[index].IsConstant && typeInstance.Values[index].IsList)
					return typeInstance.Values[index].List.Items;
		throw new InvalidOperationException("Cannot split loop value " + value + " into variables");
	}

	private static int GetLength(ValueInstance iterableInstance)
	{
		if (iterableInstance.IsText)
			return iterableInstance.Text.Length;
		if (iterableInstance.IsList)
			return iterableInstance.List.Count;
		return (int)iterableInstance.Number;
	}

	private void AlterValueVariable(ValueInstance iterableVariable,
		LoopBeginInstruction loopBegin)
	{
		var frame = Memory.Frame;
		var index = (int)frame.Get(IndexSymbolId).Number;
		if (iterableVariable.IsText)
		{
			if (index < iterableVariable.Text.Length)
				frame.Set(ValueSymbolId,
					new ValueInstance(iterableVariable.Text[index].ToString()), true,
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
		frame.Set(ValueSymbolId,
			new ValueInstance(executable.numberType, index + 1), true, Type.ValueLowercase);
	}

	private void TryStoreInstructions(Instruction instruction)
	{
		if (instruction.InstructionType == InstructionType.Set)
		{
			var set = (SetInstruction)instruction;
			Memory.Registers[set.Register] = CloneConstantValue(set.ValueInstance);
		}
		else if (instruction.InstructionType == InstructionType.StoreConstantToVariable)
		{
			var storeVariable = (StoreVariableInstruction)instruction;
			var value = CloneConstantValue(storeVariable.ValueInstance);
			StoreIdentifierValue(storeVariable.Identifier, value, storeVariable.IsMember);
		}
		else if (instruction.InstructionType == InstructionType.StoreRegisterToVariable)
		{
			var storeFromRegister = (StoreFromRegisterInstruction)instruction;
			if (!TryStoreToListElement(storeFromRegister))
				StoreIdentifierValue(storeFromRegister.Identifier,
					Memory.Registers[storeFromRegister.Register], false);
		}
	}

	private void TryLoadInstructions(Instruction instruction)
	{
		if (instruction.InstructionType == InstructionType.LoadVariableToRegister)
		{
			var loadVariable = (LoadVariableToRegister)instruction;
			if (!GetIdentifierAccessPath(loadVariable.Identifier).TryResolve(this,
				out var registerValue))
				throw new InvalidOperationException("Could not resolve variable " + //ncrunch: no coverage
					loadVariable.Identifier);
			Memory.Registers[loadVariable.Register] = registerValue;
		}
		else if (instruction.InstructionType == InstructionType.LoadConstantToRegister)
		{
			var loadConstant = (LoadConstantInstruction)instruction;
			Memory.Registers[loadConstant.Register] = CloneConstantValue(loadConstant.Constant);
		}
	}

	private static ValueInstance CloneConstantValue(ValueInstance value) =>
		value.IsList
			? new ValueInstance(value.List.Clone(value.List.ReturnType))
			: value.IsDictionary
				? new ValueInstance(value.GetType(), new Dictionary<ValueInstance, ValueInstance>(
					value.GetDictionaryItems()))
				: value;

	private IdentifierAccessPath GetIdentifierAccessPath(string identifier)
	{
		//TODO: remove this
		if (identifier.Length == 0 || double.TryParse(identifier, out _))
			return default;

		return identifierAccessPaths.TryGetValue(identifier, out var accessPath)
			? accessPath
			: identifierAccessPaths[identifier] = IdentifierAccessPath.Parse(identifier);
	}

	private bool TryGetFrameValue(int symbolId, out ValueInstance value) =>
		Memory.Frame.TryGet(symbolId, out value);

	private void StoreIdentifierValue(string identifier, ValueInstance value, bool isMember)
	{
		var accessPath = GetIdentifierAccessPath(identifier);
		if (accessPath.MemberNames.Length == 0)
		{
			Memory.Frame.Set(accessPath.RootSymbolId, value, isMember, identifier);
			return;
		}
		if (!accessPath.GetParentPath().TryResolve(this, out var parentValue))
			throw new InvalidOperationException("Could not resolve parent for " + identifier);
		var memberName = accessPath.MemberNames[^1];
		var flatInstance = parentValue.TryGetFlatNumericArrayInstance();
		if (flatInstance != null)
		{
			if (!flatInstance.TrySetMember(memberName, value))
				throw new InvalidOperationException("Could not assign member " + identifier);
			return;
		}
		if (parentValue.TryGetValueTypeInstance() is not { } typeInstance)
			throw new InvalidOperationException("Cannot assign member on non-instance " + identifier);
		if (!typeInstance.TrySetValue(memberName, value))
			throw new InvalidOperationException("Could not assign member " + identifier);
	}

	private ValueInstance TryGetNativeMemberValue(ValueInstance current, string memberName) =>
		current.IsText && memberName is "characters" or Type.ElementsLowercase
			? current
			: memberName is "Length"
				? current.IsText
					? new ValueInstance(executable.numberType, current.Text.Length)
					: current.IsList
						? new ValueInstance(executable.numberType, current.List.Count)
						: default
				: default;

	private void ExecuteBinaryInstruction(BinaryInstruction instruction)
	{
		if (instruction.IsConditional())
			ExecuteConditionalOperation(instruction);
		else
			ExecuteBinaryOperation(instruction);
	}

	private void ExecuteBinaryOperation(BinaryInstruction instruction)
	{
		var (right, left) = GetOperands(instruction);
		Memory.Registers[instruction.Registers[^1]] = instruction.InstructionType switch
		{
			InstructionType.Add => AddValueInstances(left, right),
			InstructionType.Subtract => SubtractValueInstances(left, right),
			InstructionType.Multiply => new ValueInstance(right.GetType(),
				left.Number * right.Number),
			InstructionType.Divide => new ValueInstance(right.GetType(),
				left.Number / right.Number),
			InstructionType.Modulo => new ValueInstance(right.GetType(),
				left.Number % right.Number),
			_ => throw new InvalidOperationException("Unsupported binary operation: " + instruction.InstructionType) //ncrunch: no coverage
		};
	}

	private static ValueInstance AddValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
		{
			left.List.Items.Add(right);
			return left;
		}
		if (right.IsList)
		{
			right.List.Items.Add(left);
			return right;
		}
		if (left.IsText || right.IsText)
			return new ValueInstance(ConvertToText(left).Text + ConvertToText(right).Text);
		return new ValueInstance(right.GetType(), left.Number + right.Number);
	}

	private static ValueInstance SubtractValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
		{
			var items = new List<ValueInstance>(left.List.Items);
			var removeIndex = items.FindIndex(item => item.Equals(right));
			if (removeIndex >= 0)
				items.RemoveAt(removeIndex);
			return new ValueInstance(left.List.ReturnType, items.ToArray());
		}
		if (left.IsText || right.IsText)
			throw new NotSupportedException("Texts cannot be subtracted: " + left + " - " + right); //ncrunch: no coverage
		return new ValueInstance(left.GetType(), left.Number - right.Number);
	}

	private (ValueInstance, ValueInstance) GetOperands(BinaryInstruction instruction) =>
		instruction.Registers.Length < 2
			? throw new OperandsRequired()
			: (Memory.Registers[instruction.Registers[1]], Memory.Registers[instruction.Registers[0]]);

	private void ExecuteConditionalOperation(BinaryInstruction instruction)
	{
		var (right, left) = GetOperands(instruction);
		conditionFlag = instruction.InstructionType switch
		{
			InstructionType.GreaterThan => left.Number > right.Number,
			InstructionType.LessThan => left.Number < right.Number,
			InstructionType.Equal => left.Equals(right),
			InstructionType.NotEqual => !left.Equals(right),
			_ => throw new InvalidOperationException("Unsupported conditional operation: " + instruction.InstructionType) //ncrunch: no coverage
		};
	}

	private void TryJumpOperation(Jump instruction)
	{
		if (conditionFlag && instruction.InstructionType is InstructionType.JumpIfTrue ||
			!conditionFlag && instruction.InstructionType is InstructionType.JumpIfFalse)
			instructionIndex += instruction.InstructionsToSkip;
	}

	private void TryJumpIfOperation(JumpIfNotZero instruction)
	{
		if (Memory.Registers[instruction.Register].Number > 0)
			instructionIndex += instruction.InstructionsToSkip;
	}

	private void TryJumpToIdOperation(JumpToId instruction)
	{
		if (!conditionFlag && instruction.InstructionType is InstructionType.JumpToIdIfFalse ||
			conditionFlag && instruction.InstructionType is InstructionType.JumpToIdIfTrue)
		{
			var endIndex = FindJumpEndInstructionIndex(instruction.Id);
			if (endIndex != -1)
				instructionIndex = endIndex;
		}
	}

	private int FindJumpEndInstructionIndex(int id)
	{
		for (var index = 0; index < instructions.Count; index++)
			if (instructions[index].InstructionType == InstructionType.JumpEnd &&
				((JumpToId)instructions[index]).Id == id)
				return index;
		return -1; //ncrunch: no coverage
	}

	public sealed class OperandsRequired : Exception;

	private bool TryStoreToListElement(StoreFromRegisterInstruction store)
	{
		var indexedAccessPath = GetIndexedElementAccessPath(store.Identifier);
		if (!indexedAccessPath.IsValid)
			return false;
		var listValue = TryResolveListValue(indexedAccessPath.ListPath);
		if (!listValue.IsList)
			return false;
		var indexInstance = TryResolveIndexValue(indexedAccessPath.IndexExpression);
		if (!indexInstance.HasValue)
			return false;
		var index = (int)indexInstance.Number;
		if (index >= 0 && index < listValue.List.Count)
		{
			listValue.List[index] = Memory.Registers[store.Register];
			return true;
		}
		return false;
	}

	private IndexedElementAccessPath GetIndexedElementAccessPath(string identifier) =>
		indexedElementAccessPaths.TryGetValue(identifier, out var accessPath)
			? accessPath
			: indexedElementAccessPaths[identifier] = IndexedElementAccessPath.Parse(identifier);

	private ValueInstance TryResolveListValue(string listPath) =>
		GetIdentifierAccessPath(listPath).TryResolve(this, out var listValue)
			? listValue
			: default;

	private ValueInstance TryResolveIndexValue(string indexExpression)
	{
		if (double.TryParse(indexExpression, out var number))
			return new ValueInstance(executable.numberType, number);
		var accessPath = GetIdentifierAccessPath(indexExpression);
		if (accessPath.TryResolve(this, out var indexInstance))
			return indexInstance;
		return TryGetFrameValue(IndexSymbolId, out indexInstance)
			? indexInstance
			: default;
	}

	private readonly record struct IdentifierAccessPath(int RootSymbolId, string[] MemberNames)
	{
		public bool TryResolve(VirtualMachine vm, out ValueInstance value)
		{
			if (MemberNames == null)
			{
				value = default;
				return false;
			}
			if (!vm.TryGetFrameValue(RootSymbolId, out var current))
			{
				value = default;
				return false;
			}
			for (var memberIndex = 0; memberIndex < MemberNames.Length; memberIndex++)
			{
				var memberName = MemberNames[memberIndex];
				if (RootSymbolId == OuterSymbolId && memberName == Type.ValueLowercase)
					continue;
				if (RootSymbolId == OuterSymbolId && memberName == Type.IndexLowercase &&
					vm.TryGetFrameValue(OuterIndexSymbolId, out var outerIndexValue))
				{
					current = outerIndexValue;
					continue;
				}
				var nativeMemberValue = vm.TryGetNativeMemberValue(current, memberName);
				if (nativeMemberValue.HasValue)
				{
					current = nativeMemberValue;
					continue;
				}
				if (current.TryGetFlatNumericMember(memberName, out var flatMember))
				{
					current = flatMember;
					continue;
				}
				var typeInstance = current.TryGetValueTypeInstance();
				if (typeInstance == null || !typeInstance.TryGetValue(memberName, out current))
				{
					value = default;
					return false;
				}
			}
			value = current;
			return true;
		}

		public static IdentifierAccessPath Parse(string identifier)
		{
			if (identifier == Type.None)
				return default;
			var firstDotIndex = identifier.IndexOf('.');
			if (firstDotIndex < 0)
				return new IdentifierAccessPath(CallFrame.ResolveSymbolId(identifier), []);
			var rootSymbolId = CallFrame.ResolveSymbolId(identifier[..firstDotIndex]);
			var memberCount = 1;
			for (var index = firstDotIndex + 1; index < identifier.Length; index++)
				if (identifier[index] == '.')
					memberCount++;
			var memberNames = new string[memberCount];
			var memberIndex = 0;
			var segmentStart = firstDotIndex + 1;
			while (segmentStart < identifier.Length)
			{
				var nextDotIndex = identifier.IndexOf('.', segmentStart);
				memberNames[memberIndex++] = nextDotIndex < 0
					? identifier[segmentStart..]
					: identifier[segmentStart..nextDotIndex];
				if (nextDotIndex < 0)
					break;
				segmentStart = nextDotIndex + 1;
			}
			return new IdentifierAccessPath(rootSymbolId, memberNames);
		}

		public IdentifierAccessPath GetParentPath() =>
			MemberNames.Length == 1
				? new IdentifierAccessPath(RootSymbolId, [])
				: new IdentifierAccessPath(RootSymbolId, MemberNames[..^1]);
	}

	private readonly record struct IndexedElementAccessPath(string ListPath, string IndexExpression,
		bool IsValid)
	{
		public static IndexedElementAccessPath Parse(string identifier)
		{
			var openParen = identifier.LastIndexOf('(');
			return openParen <= 0 || !identifier.EndsWith(')')
				? new IndexedElementAccessPath(string.Empty, string.Empty, false)
				: new IndexedElementAccessPath(identifier[..openParen], identifier[(openParen + 1)..^1],
					true);
		}
	}
}
