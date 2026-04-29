using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine(BinaryExecutable executable)
{
	public VirtualMachine(Package basePackage) : this(new BinaryExecutable(basePackage)) { }

	public VirtualMachine Execute(BinaryMethod? method = null,
		IReadOnlyDictionary<string, ValueInstance>? initialVariables = null)
	{
		method ??= executable.EntryPoint;
		conditionFlag = false;
		Returns = null;
		Memory.Registers.Clear();
		Memory.Frame = new CallFrame(initialVariables);
		InitializeEntryPointMembers(method);
		currentMethodContext = ResolveMethodContext(method);
		RunInstructions(method.instructions
#if DEBUG
			, method.Name
#endif
		);
		DisposeTrackedValues(Memory.Frame, Returns, null);
		return this;
	}

	private bool conditionFlag;
	private int instructionIndex;
	//TODO: why do we copy so many lists around, use BinaryMethod!
	private List<Instruction> instructions = [];
	public ValueInstance? Returns { get; private set; }
	public Memory Memory { get; } = new();
	private string currentMethodContext = "";
	private const int MaxCallDepth = 1024;
	private readonly ValueInstance[][] registerStack = new ValueInstance[MaxCallDepth][];
	private int registerStackDepth;
	private readonly CallFrame[] framePool = new CallFrame[MaxCallDepth];
	private int framePoolDepth;
	private static readonly int ValueSymbolId = CallFrame.ValueSymbolId;
	private static readonly int IndexSymbolId = CallFrame.IndexSymbolId;
	private static readonly int OuterSymbolId = CallFrame.OuterSymbolId;
	private static readonly int OuterIndexSymbolId =
		CallFrame.ResolveSymbolId(Type.OuterLowercase + "." + Type.IndexLowercase);
	/*TODO: remove, unused
	private static readonly int ElementsSymbolId = CallFrame.ElementsSymbolId;
	private static readonly int CharactersSymbolId = CallFrame.CharactersSymbolId;
	private readonly int noneSymbolId = CallFrame.ResolveSymbolId(Type.None);
	*/
	private readonly Dictionary<string, IdentifierAccessPath> identifierAccessPaths =
		new(StringComparer.Ordinal);
	private readonly Dictionary<string, IndexedElementAccessPath> indexedElementAccessPaths =
		new(StringComparer.Ordinal);

	private VirtualMachine RunInstructions(List<Instruction> blockInstructions
#if DEBUG
		, string context = "body"
#endif
	)
	{
#if DEBUG
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("VirtualMachine.RunInstructions",
				"context=" + context + ", count=" + blockInstructions.Count);
#endif
		CacheInstructionAccessPaths(blockInstructions);
		for (var index = 0; index < blockInstructions.Count; index++)
			if (blockInstructions[index].InstructionType == InstructionType.LoopBegin)
			{
				var loopBegin = (LoopBeginInstruction)blockInstructions[index];
				loopBegin.Reset();
				loopBegin.InstructionIndex = index;
			}
		instructions = blockInstructions;
		var instructionsLength = instructions.Count;
		for (instructionIndex = 0; instructionIndex < instructionsLength; instructionIndex++)
			ExecuteInstruction(instructions[instructionIndex]);
		return this;
	}

	private void CacheInstructionAccessPaths(List<Instruction> blockInstructions)
	{
		identifierAccessPaths.EnsureCapacity(
			identifierAccessPaths.Count + blockInstructions.Count * 2);
		indexedElementAccessPaths.EnsureCapacity(indexedElementAccessPaths.Count +
			blockInstructions.Count);
		for (var cachedInstructionIndex = 0; cachedInstructionIndex < blockInstructions.Count;
			cachedInstructionIndex++)
			switch (blockInstructions[cachedInstructionIndex])
			{
			case LoadVariableToRegister loadVariable:
				GetIdentifierAccessPath(loadVariable.Identifier);
				break;
			case StoreVariableInstruction storeVariable:
				GetIdentifierAccessPath(storeVariable.Identifier);
				break;
			case StoreFromRegisterInstruction storeFromRegister:
				CacheStoreAccessPath(storeFromRegister.Identifier);
				break;
			case ListCallInstruction listCall:
				GetIdentifierAccessPath(listCall.Identifier);
				break;
			case WriteToListInstruction writeToList:
				GetIdentifierAccessPath(writeToList.Identifier);
				break;
			case RemoveInstruction remove:
				GetIdentifierAccessPath(remove.Identifier);
				break;
			}
	}

	private void CacheStoreAccessPath(string identifier)
	{
		var indexedAccessPath = GetIndexedElementAccessPath(identifier);
		if (indexedAccessPath.IsValid)
		{
			GetIdentifierAccessPath(indexedAccessPath.ListPath);
			GetIdentifierAccessPath(indexedAccessPath.IndexExpression);
		}
		else
			GetIdentifierAccessPath(identifier);
	}

	private void InitializeEntryPointMembers(BinaryMethod method)
	{
		foreach (var type in executable.MethodsPerType)
		foreach (var overloads in type.Value.MethodGroups.Values)
			if (overloads.Contains(method))
			{
				foreach (var member in type.Value.Members)
					Memory.Frame.Set(member.Name,
						member.InitialValueExpression is SetInstruction setInstruction
							? CloneConstantValue(setInstruction.ValueInstance)
							: CreateDefaultComplexValue(ResolveBinaryMemberType(member, type.Key)),
						isMember: true);
				return;
			}
	}

	private Type ResolveBinaryMemberType(BinaryMember member, string entryTypeName)
	{
		var fullTypeName = member.FullTypeName;
		var contextualTypeName = GetBinaryMemberContextualTypeName(entryTypeName, fullTypeName);
		return (fullTypeName.Contains(Context.ParentSeparator)
				? executable.basePackage.FindFullType(fullTypeName)
				: null) ?? executable.basePackage.FindType(fullTypeName) ??
			executable.basePackage.FindType(member.JustTypeName) ??
			executable.basePackage.FindType(contextualTypeName) ?? executable.numberType;
	}

	private static string GetBinaryMemberContextualTypeName(string entryTypeName,
		string memberTypeName)
	{
		var separatorIndex = entryTypeName.LastIndexOf(Context.ParentSeparator);
		return separatorIndex < 0
			? memberTypeName
			: string.Concat(entryTypeName.AsSpan(0, separatorIndex + 1), memberTypeName);
	}

	//TODO: almost called 3 million times, our loop is only 0.23m, so we are doing too much each time
	//TODO: inlining should reduce the call to GetBrightnessAdjustedColor to something much simpler
	//for colorIndex in Range(0, image.Colors.Length)
	//	image.Colors(colorIndex) = GetBrightnessAdjustedColor(image.Colors(colorIndex))
	//+GetBrightnessAdjustedColor(current Color) Color
	//+ Color(current.Red + brightness, current.Green + brightness, current.Blue + brightness)
	//should be:
	//for colorIndex in Range(0, image.Colors.Length)
	//	keep this color, maybe as value?: image.Colors(colorIndex)
	//  color.Red += brightness (e.g. using color.Red.Increase(brightness))
	//  color.Green += brightness
	//  color.Blue += brightness
	// Now it should be 0.23m*(3-4) instructions, less than 1m, also no lookups, we can keep value and brightness directly in memory and reuse, index increases were we are in our big Colors array ..
	private void ExecuteInstruction(Instruction instruction)
	{
#if DEBUG
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("** VirtualMachine.ExecuteInstruction",
				"index=" + instructionIndex + ", type=" + instruction.InstructionType +
				GetInstructionDetails(instruction));
#endif
		switch (instruction.InstructionType)
		{
		case InstructionType.Return:
			ExecuteReturn((ReturnInstruction)instruction);
			return;
		case InstructionType.Set:
		case InstructionType.StoreConstantToVariable:
		case InstructionType.StoreRegisterToVariable:
			TryStoreInstructions(instruction);
			return;
		case InstructionType.LoadVariableToRegister:
		case InstructionType.LoadConstantToRegister:
			TryLoadInstructions(instruction);
			return;
		case InstructionType.LoopBegin:
			ExecuteLoopBegin((LoopBeginInstruction)instruction);
			return;
		case InstructionType.LoopEnd:
			ExecuteLoopEnd((LoopEndInstruction)instruction);
			return;
		case InstructionType.Print:
			ExecutePrint((PrintInstruction)instruction);
			return;
		case InstructionType.FieldLoad:
			ExecuteFieldLoad((FieldLoadInstruction)instruction);
			return;
		case InstructionType.ConstructValueType:
			ExecuteConstructValueType((ConstructValueTypeInstruction)instruction);
			return;
		case InstructionType.Invoke:
			ExecuteInvoke((Invoke)instruction);
			return;
		case InstructionType.InvokeWriteToList:
			ExecuteWriteToList((WriteToListInstruction)instruction);
			return;
		case InstructionType.InvokeWriteToTable:
			ExecuteWriteToTable((WriteToTableInstruction)instruction);
			return;
		case InstructionType.InvokeRemove:
			ExecuteRemove((RemoveInstruction)instruction);
			return;
		case InstructionType.ListCall:
			ExecuteListCall((ListCallInstruction)instruction);
			return;
		case InstructionType.Jump:
		case InstructionType.JumpIfTrue:
		case InstructionType.JumpIfFalse:
			TryJumpOperation((Jump)instruction);
			return;
		case InstructionType.JumpIfNotZero:
			TryJumpIfOperation((JumpIfNotZero)instruction);
			return;
		case InstructionType.JumpEnd:
		case InstructionType.JumpToIdIfFalse:
		case InstructionType.JumpToIdIfTrue:
			TryJumpToIdOperation((JumpToId)instruction);
			return;
		case InstructionType.Add:
		case InstructionType.Subtract:
		case InstructionType.Multiply:
		case InstructionType.Divide:
		case InstructionType.Modulo:
		case InstructionType.Equal:
		case InstructionType.NotEqual:
		case InstructionType.LessThan:
		case InstructionType.GreaterThan:
			ExecuteBinaryInstruction((BinaryInstruction)instruction);
			return;
		default:
			throw new InvalidInstruction(instruction); //ncrunch: no coverage
		}
	}
#if DEBUG
	private static string GetInstructionDetails(Instruction instruction) =>
		instruction switch
		{
			StoreVariableInstruction storeVariable => ", Identifier=" + storeVariable.Identifier +
				", IsMember=" + storeVariable.IsMember + ", ValueInstance=" +
				DescribeValueInstance(storeVariable.ValueInstance),
			StoreFromRegisterInstruction storeFromRegister => ", Identifier=" +
				storeFromRegister.Identifier + ", Register=" + storeFromRegister.Register,
			LoadVariableToRegister loadVariable => ", Identifier=" + loadVariable.Identifier +
				", Register=" + loadVariable.Register,
			LoadConstantInstruction loadConstant => ", Constant=" +
				DescribeValueInstance(loadConstant.Constant) + ", Register=" + loadConstant.Register,
			Invoke invoke => ", Method=" + invoke.MethodInfo + ", Register=" +
				invoke.Register,
			PrintInstruction print => ", TextPrefix=" + print.TextPrefix + ", ValueRegister=" +
				print.ValueRegister + ", ValueIsText=" + print.ValueIsText,
			LoopBeginInstruction loopBegin => ", Register=" + loopBegin.Register + ", IsRange=" +
				loopBegin.IsRange + ", CustomVariableNames=" +
				string.Join(", ", loopBegin.CustomVariableNames),
			LoopEndInstruction loopEnd => ", Steps=" + loopEnd.Steps,
			BinaryInstruction binary => ", Registers=" + DescribeRegisters(binary.Registers),
			ListCallInstruction listCall => ", Identifier=" + listCall.Identifier + ", Register=" +
				listCall.Register + ", IndexValueRegister=" + listCall.IndexValueRegister,
			WriteToListInstruction writeToList => ", Identifier=" + writeToList.Identifier +
				", Register=" + writeToList.Register,
			WriteToTableInstruction writeToTable => ", Identifier=" + writeToTable.Identifier +
				", KeyRegister=" + writeToTable.Register + ", ValueRegister=" + writeToTable.Value,
			RemoveInstruction remove => ", Identifier=" + remove.Identifier + ", Register=" +
				remove.Register,
			FieldLoadInstruction fieldLoad => ", FieldName=" + fieldLoad.FieldName +
				", ObjectRegister=" + fieldLoad.ObjectRegister + ", Register=" + fieldLoad.Register,
			ConstructValueTypeInstruction construct => ", ReturnType=" + construct.ReturnType.Name +
				", Register=" + construct.Register + ", Fields=" +
				DescribeRegisters(construct.FieldRegisters),
			JumpIfNotZero jumpIfNotZero => ", Register=" + jumpIfNotZero.Register +
				", InstructionsToSkip=" + jumpIfNotZero.InstructionsToSkip,
			JumpToId jumpToId => ", Id=" + jumpToId.Id,
			Jump jump => ", InstructionsToSkip=" + jump.InstructionsToSkip,
			_ => string.Empty
		};

	private static string DescribeRegisters(Register[] registers)
	{
		if (registers.Length == 0)
			return "[]";
		var parts = new string[registers.Length];
		for (var index = 0; index < registers.Length; index++)
			parts[index] = registers[index].ToString();
		return "[" + string.Join(", ", parts) + "]";
	}

	private static string DescribeValueInstance(ValueInstance value) =>
		!value.HasValue
			? "unset"
			: value.IsText
				? "Text(length=" + value.Text.Length + ")"
				: value.IsList
					? "List(type=" + value.List.ReturnType.Name + ", count=" + value.List.Items.Count + ")"
					: value.IsDictionary
						? "Dictionary(count=" + value.GetDictionaryItems().Count + ")"
						: value.TryGetValueTypeInstance() is { } typeInstance
							? "TypeInstance(type=" + typeInstance.ReturnType.Name + ", members=" +
							typeInstance.Values.Length + ")"
							: value.GetType().Name + "(" + value.Number + ")";
#endif
	private sealed class InvalidInstruction(Instruction instruction)
		: Exception(instruction.ToString()); //ncrunch: no coverage

	public sealed class StackOverflow(int maxDepth)
		: Exception("VM call depth exceeded limit: " + maxDepth);

	private string ResolveMethodContext(BinaryMethod method)
	{
		foreach (var (typeFullName, typeData) in executable.MethodsPerType)
		foreach (var overloads in typeData.MethodGroups.Values)
			if (overloads.Contains(method))
				return typeFullName + "." + method.Name;
		return method.Name;
	}

	private InstructionExecutionFailed Fail(string message, Exception? inner = null)
	{
		var index = Math.Max(0, Math.Min(instructionIndex, instructions.Count - 1));
		var (sourceLines, filePath) = TryGetSourceContext();
		return new InstructionExecutionFailed(message, instructions, index, currentMethodContext,
			sourceLines, filePath, inner);
	}

	private (string[]? lines, string filePath) TryGetSourceContext()
	{
		var dotIndex = currentMethodContext.LastIndexOf('.');
		if (dotIndex < 0)
			return (null, "");
		var typeFullName = currentMethodContext[..dotIndex];
		var type = typeFullName.Contains('/')
			? executable.basePackage.FindFullType(typeFullName)
			: executable.basePackage.FindType(typeFullName);
		if (type == null)
			return (null, "");
		var filePath = type.FilePath;
		return File.Exists(filePath)
			? (File.ReadAllLines(filePath), filePath)
			: (null, filePath);
	}

	private void ExecuteFieldLoad(FieldLoadInstruction instr)
	{
		var typeInstance = Memory.Registers[instr.ObjectRegister].TryGetValueTypeInstance()!;
		var members = typeInstance.ReturnType.Members;
		for (var index = 0; index < members.Count; index++)
			if (members[index].Name.Equals(instr.FieldName, StringComparison.OrdinalIgnoreCase))
			{
				Memory.Registers[instr.Register] = typeInstance.Values[index];
				return;
			}
	}

	private void ExecuteConstructValueType(ConstructValueTypeInstruction instr)
	{
		var members = instr.ReturnType.Members;
		var values = new ValueInstance[members.Count];
		for (var index = 0; index < instr.FieldRegisters.Length && index < members.Count; index++)
			values[index] = Memory.Registers[instr.FieldRegisters[index]];
		for (var index = instr.FieldRegisters.Length; index < members.Count; index++)
			values[index] = members[index].Type.IsTrait
				? CreateTraitInstance(members[index].Type)
				: members[index].InitialValue is Value initialValue
					? initialValue.Data
					: CreateDefaultValue(members[index].Type);
		TryPreFillConstrainedListMembers(instr.ReturnType, values);
		Memory.Registers[instr.Register] = new ValueInstance(instr.ReturnType, values);
	}

	private void ExecutePrint(PrintInstruction print)
	{
		if (print.ValueRegister.HasValue)
			Console.WriteLine(print.TextPrefix +
				Memory.Registers[print.ValueRegister.Value].ToExpressionCodeString());
		else
			Console.WriteLine(print.TextPrefix);
	}

	private void ExecuteRemove(RemoveInstruction removeInstruction)
	{
		var item = Memory.Registers[removeInstruction.Register];
		var items = Memory.Frame.Get(removeInstruction.Identifier).List.Items;
		//TODO: is there actually a need for this loop? or is there always 1 entry anyway?
		for (var itemIndex = items.Count - 1; itemIndex >= 0; itemIndex--)
			if (items[itemIndex].Equals(item))
				items.RemoveAt(itemIndex);
	}

	private void ExecuteListCall(ListCallInstruction listCallInstruction)
	{
		var indexValue = (int)Memory.Registers[listCallInstruction.IndexValueRegister].Number;
		var collectionValue = Memory.Frame.Get(listCallInstruction.Identifier);
		if (collectionValue is { IsList: false, IsText: false })
		{
			if (listCallInstruction.Identifier == Type.OuterLowercase &&
				TryGetFrameValue(ValueSymbolId, out var currentValue))
				collectionValue = currentValue;
			else if (listCallInstruction.Identifier is "characters" or Type.ElementsLowercase &&
				TryGetFrameValue(OuterSymbolId, out var outerValue) && outerValue.IsText)
				collectionValue = outerValue;
		}
		if (collectionValue.IsText)
		{
			Memory.Registers[listCallInstruction.Register] =
				new ValueInstance(collectionValue.Text[indexValue].ToString());
			return;
		}
		if (!collectionValue.IsList)
			throw Fail("Cannot index non-list variable \"" +
				listCallInstruction.Identifier + "\" of type " + collectionValue.GetType().Name);
		if (indexValue < 0 || indexValue >= collectionValue.List.Count)
			throw Fail("List index out of range for \"" + listCallInstruction.Identifier +
				"\": " + indexValue + " (count=" + collectionValue.List.Count + ")");
		var variableListElement = collectionValue.List[indexValue];
		Memory.Registers[listCallInstruction.Register] = variableListElement;
	}

	private void ExecuteWriteToList(WriteToListInstruction writeToListInstruction)
	{
		if (!GetIdentifierAccessPath(writeToListInstruction.Identifier).TryResolve(this, out var collection))
			throw Fail("Cannot resolve list variable \"" + writeToListInstruction.Identifier + "\"");
		collection.List.Items.Add(Memory.Registers[writeToListInstruction.Register]);
	}

	private void ExecuteWriteToTable(WriteToTableInstruction writeToTableInstruction)
	{
		if (!GetIdentifierAccessPath(writeToTableInstruction.Identifier).TryResolve(this, out var collection))
			throw Fail("Cannot resolve table variable \"" + writeToTableInstruction.Identifier + "\"");
		collection.GetDictionaryItems()[Memory.Registers[writeToTableInstruction.Register]] =
			Memory.Registers[writeToTableInstruction.Value];
	}

	private void ExecuteLoopEnd(LoopEndInstruction loopEnd)
	{
		var loopBegin = loopEnd.Begin ?? FindLoopBeginByScanning(loopEnd.Steps);
		loopBegin.LoopCount--;
		if (loopBegin.LoopCount <= 0)
		{
			RestoreLoopState(loopBegin, Memory.Frame);
			return;
		}
		instructionIndex = loopBegin.InstructionIndex - 1;
	}

	/// <summary>
	/// Fallback for deserialized LoopEndInstructions that don't have LoopBegin set.
	/// Uses Steps as a hint to find the LoopBeginInstruction by scanning.
	/// </summary>
	private LoopBeginInstruction FindLoopBeginByScanning(int steps)
	{
		var loopDepth = 0;
		for (var candidateIndex = instructionIndex - 1;
			candidateIndex >= Math.Max(0, instructionIndex - steps - 1); candidateIndex--)
			if (instructions[candidateIndex].InstructionType == InstructionType.LoopEnd)
				loopDepth++;
			else if (instructions[candidateIndex].InstructionType == InstructionType.LoopBegin)
				if (loopDepth == 0)
					return (LoopBeginInstruction)instructions[candidateIndex];
				else
					loopDepth--;
		throw Fail("No matching LoopBeginInstruction found for LoopEnd");
	}

	private void ExecuteReturn(ReturnInstruction returnInstruction)
	{
		Returns = Memory.Registers[returnInstruction.Register];
		instructionIndex = ExitExecutionLoopIndex;
	}

	private const int ExitExecutionLoopIndex = 100_000;
}
