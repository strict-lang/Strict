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
		return RunInstructions(method.instructions
#if DEBUG
			, method.Name
#endif
		);
	}

	private bool conditionFlag;
	private int instructionIndex;
	//TODO: why do we copy so many lists around, use BinaryMethod!
	private List<Instruction> instructions = [];
	public ValueInstance? Returns { get; private set; }
	public Memory Memory { get; } = new();
	private string currentMethodContext = "";
	private const int MaxCallDepth = 64;
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
		var type = executable.basePackage.FindFullType(typeFullName) ??
			executable.basePackage.FindType(typeFullName.Contains('/')
				? typeFullName[(typeFullName.LastIndexOf('/') + 1)..]
				: typeFullName);
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
		var idx = Math.Max(0, instructionIndex - steps);
		while (idx < instructions.Count && instructions[idx] is not LoopBeginInstruction)
			idx++;
		return idx < instructions.Count
			? (LoopBeginInstruction)instructions[idx]
			: throw new InvalidOperationException("No matching LoopBeginInstruction found for LoopEnd");
	}

	private void TryInvokeInstruction(Instruction instruction)
	{
		if (instruction is not Invoke invoke ||
			TryCreateEmptyDictionaryInstance(invoke) || TryHandleFromConstructor(invoke) ||
			TryHandleNativeTraitMethod(invoke) || TryHandleToConversion(invoke) ||
			TryHandleIncrementDecrement(invoke) || GetValueByKeyForDictionaryAndStoreInRegister(invoke) ||
			TryHandleNativeTextMethod(invoke))
			return;
		var evaluatedArgs = invoke.Method.Arguments.Select(EvaluateExpression).ToArray();
		var evaluatedInstance = invoke.Method.Instance != null
			? EvaluateExpression(invoke.Method.Instance)
			: (ValueInstance?)null;
		var invokeInstructions = GetPrecompiledMethodInstructions(invoke) ??
			throw new InvalidOperationException("No precompiled method instructions found for invoke");
		var result = RunChildScope(invokeInstructions,
			() => InitializeMethodCallScope(invoke.Method, evaluatedArgs, evaluatedInstance));
		if (result != null)
			Memory.Registers[invoke.Register] = result.Value;
	}

	//TODO: find all [.. with existing list and no changes, all those cases need to be removed, there is a crazy amount of those added (54 wtf)!
	private IReadOnlyList<Instruction>? GetPrecompiledMethodInstructions(Method method) =>
		executable.FindInstructions(method.Type, method) ??
		executable.FindInstructions(method.Type.Name, method.Name,
			method.Parameters.Count, method.ReturnType.Name) ??
		executable.FindInstructions(
			nameof(Strict) + Context.ParentSeparator + method.Type.Name, method.Name,
			method.Parameters.Count, method.ReturnType.Name);

	private IReadOnlyList<Instruction>? GetPrecompiledMethodInstructions(Invoke invoke) =>
		GetPrecompiledMethodInstructions(invoke.Method.Method);

	private void InitializeMethodCallScope(MethodCall methodCall,
		IReadOnlyList<ValueInstance>? evaluatedArguments = null,
		ValueInstance? evaluatedInstance = null)
	{
		for (var parameterIndex = 0; parameterIndex < methodCall.Method.Parameters.Count &&
			parameterIndex < methodCall.Arguments.Count; parameterIndex++)
			Memory.Frame.Set(methodCall.Method.Parameters[parameterIndex].Name,
				evaluatedArguments != null
					? evaluatedArguments[parameterIndex]
					: EvaluateExpression(methodCall.Arguments[parameterIndex]));
		if (methodCall.Instance == null)
			return;
		var instance = evaluatedInstance ?? EvaluateExpression(methodCall.Instance);
		Memory.Frame.Set(Type.ValueLowercase, instance, isMember: true);
		if (instance.IsText)
		{
			Memory.Frame.Set("elements", instance, isMember: true);
			Memory.Frame.Set("characters", instance, isMember: true);
			return;
		}
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null && (TrySetScopeMembersFromTypeMembers(typeInstance) ||
			TrySetScopeMembersFromBinaryMembers(typeInstance)))
			return;
		var firstNonTraitMember = instance.GetType().Members.FirstOrDefault(member =>
			!member.Type.IsTrait);
		if (firstNonTraitMember != null)
			Memory.Frame.Set(firstNonTraitMember.Name, instance, isMember: true);
	}

	private bool TrySetScopeMembersFromTypeMembers(ValueTypeInstance typeInstance)
	{
		var members = typeInstance.ReturnType.Members;
		if (members.Count == 0)
			return false;
		for (var memberIndex = 0; memberIndex < members.Count &&
			memberIndex < typeInstance.Values.Length; memberIndex++)
			if (!members[memberIndex].Type.IsTrait)
				Memory.Frame.Set(members[memberIndex].Name, typeInstance.Values[memberIndex],
					isMember: true);
		return true;
	}

	private bool TrySetScopeMembersFromBinaryMembers(ValueTypeInstance typeInstance)
	{
		if (!TryGetBinaryMembers(typeInstance.ReturnType, out var binaryMembers))
			return false;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count &&
			memberIndex < typeInstance.Values.Length; memberIndex++)
			if (CanExposeBinaryMember(typeInstance.ReturnType, binaryMembers[memberIndex]))
				Memory.Frame.Set(binaryMembers[memberIndex].Name, typeInstance.Values[memberIndex],
					isMember: true);
		return true;
	}

	private bool TryGetBinaryMembers(Type type, out IReadOnlyList<BinaryMember> members)
	{
		foreach (var (typeName, typeData) in executable.MethodsPerType)
			if (typeData.Members.Count > 0 && (typeName == type.FullName || typeName == type.Name ||
				typeName.EndsWith(Context.ParentSeparator + type.Name, StringComparison.Ordinal)))
			{
				members = typeData.Members;
				return true;
			}
		members = [];
		return false;
	}

	private static bool CanExposeBinaryMember(Type instanceType, BinaryMember binaryMember)
	{
		var memberType = instanceType.FindType(binaryMember.FullTypeName) ??
			instanceType.FindType(GetShortTypeName(binaryMember.FullTypeName));
		return memberType == null || !memberType.IsTrait;
	}

	private static string GetShortTypeName(string fullTypeName)
	{
		var index = fullTypeName.LastIndexOf(Context.ParentSeparator);
		return index >= 0
			? fullTypeName[(index + 1)..]
			: fullTypeName;
	}

	private ValueInstance? RunChildScope(IReadOnlyList<Instruction> childInstructions,
		Action? initializeScope = null)
	{
		var savedInstructions = instructions;
		var savedIndex = instructionIndex;
		var savedConditionFlag = conditionFlag;
		var savedReturns = Returns;
		var savedFrame = Memory.Frame;
		var savedRegisters = Memory.Registers.Save();
		Memory.Frame = new CallFrame(savedFrame);
		initializeScope?.Invoke();
		Returns = null;
		RunInstructions(childInstructions);
		var result = Returns;
		Memory.Frame.Clear();
		Memory.Frame = savedFrame;
		Memory.Registers.Restore(savedRegisters);
		instructions = savedInstructions;
		instructionIndex = savedIndex;
		conditionFlag = savedConditionFlag;
		Returns = savedReturns;
		return result;
	}

	private bool TryHandleIncrementDecrement(Invoke invoke)
	{
		var methodName = invoke.Method.Method.Name;
		if (methodName != "Increment" && methodName != "Decrement")
			return false;
		if (invoke.Method.Instance == null ||
			!Memory.Frame.TryGet(invoke.Method.Instance.ToString(), out var current))
			return false;
		var delta = methodName == "Increment"
			? 1.0
			: -1.0;
		Memory.Registers[invoke.Register] =
			new ValueInstance(current.GetType(), current.Number + delta);
		return true;
	}

	private bool TryHandleNativeTextMethod(Invoke invoke)
	{
		var methodName = invoke.Method.Method.Name;
		if (methodName is not ("StartsWith" or "IndexOf" or "Substring"))
			return false;
		if (invoke.Method.Instance == null)
			return false;
		var instance = EvaluateExpression(invoke.Method.Instance);
		if (!instance.IsText)
			return false;
		var text = instance.Text;
		var args = invoke.Method.Arguments.Select(EvaluateExpression).ToArray();
		Memory.Registers[invoke.Register] = methodName switch
		{
			"StartsWith" => EvaluateStartsWith(text, args),
			"IndexOf" => new ValueInstance(executable.numberType,
				text.IndexOf(args[0].Text, StringComparison.Ordinal)),
			"Substring" => new ValueInstance(
				text.Substring((int)args[0].Number, (int)args[1].Number)),
			_ => throw new InvalidOperationException("Unhandled native text method: " + methodName)
		};
		return true;
	}

	private ValueInstance EvaluateStartsWith(string text, ValueInstance[] args)
	{
		var prefix = args[0].Text;
		var start = args.Length > 1
			? (int)args[1].Number
			: 0;
		var matches = start + prefix.Length <= text.Length &&
			text.AsSpan(start, prefix.Length).SequenceEqual(prefix);
		return new ValueInstance(executable.booleanType, matches
			? 1.0
			: 0.0);
	}

	private bool TryHandleToConversion(Invoke invoke)
	{
		if (invoke.Method.Method.Name != BinaryOperator.To)
			return false;
		var instanceExpr = invoke.Method.Instance ?? throw new InvalidOperationException();
		var rawValue = instanceExpr is Value constValue
			? constValue.Data
			: Memory.Frame.TryGet(instanceExpr.ToString(), out var variableValue)
				? variableValue
				: throw new InvalidOperationException();
		var conversionType = invoke.Method.ReturnType;
		if (conversionType.IsText)
			Memory.Registers[invoke.Register] = ConvertToText(rawValue);
		else if (conversionType.IsNumber)
			Memory.Registers[invoke.Register] =
				rawValue.IsText
					? new ValueInstance(conversionType, Convert.ToDouble(rawValue.Text))
					: rawValue;
		return true;
	}

	private static ValueInstance ConvertToText(ValueInstance rawValue)
	{
		if (rawValue.IsText)
			return rawValue;
		if (rawValue.TryGetValueTypeInstance() is { } typeInstance)
		{
			var members = typeInstance.ReturnType.Members;
			var memberValues = new List<string>(typeInstance.Values.Length);
			for (var memberIndex = 0; memberIndex < typeInstance.Values.Length && memberIndex < members.Count; memberIndex++)
				if (!members[memberIndex].Type.IsTrait && members[memberIndex].Type.Name is not
					(Type.Logger or Type.TextWriter or Type.System))
					memberValues.Add(typeInstance.Values[memberIndex].ToExpressionCodeString());
			return memberValues.Count == 0
				? new ValueInstance(typeInstance.ReturnType.Name)
				: new ValueInstance("(" + string.Join(", ", memberValues) + ")");
		}
		return new ValueInstance(rawValue.ToExpressionCodeString());
	}

	private bool TryCreateEmptyDictionaryInstance(Invoke invoke)
	{
		if (invoke.Method.Instance != null || invoke.Method.Method.Name != Method.From ||
			invoke.Method.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Type.Dictionary
			} dictionaryType)
			return false;
		Memory.Registers[invoke.Register] = new ValueInstance(dictionaryType, new Dictionary<ValueInstance, ValueInstance>());
		return true;
	}

	/// <summary>
	/// Handles From constructor calls like SimpleCalculator(2, 3) by creating a ValueInstance
	/// with evaluated argument values for each non-trait member.
	/// </summary>
	private bool TryHandleFromConstructor(Invoke invoke)
	{
		if (invoke.Method.Method.Name != Method.From || invoke.Method.Instance != null)
			return false;
		var targetType = invoke.Method.ReturnType;
		if (targetType is GenericTypeImplementation)
			return false;
		var members = targetType.Members;
		if (members.Count == 0 && TryGetBinaryMembers(targetType, out var binaryMembers))
		{
			Memory.Registers[invoke.Register] = new ValueInstance(targetType,
				CreateConstructorValuesFromBinaryMembers(targetType, invoke, binaryMembers));
			return true;
		}
		var values = new ValueInstance[members.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (members[memberIndex].Type.IsTrait)
				values[memberIndex] = CreateTraitInstance(members[memberIndex].Type);
			else if (argumentIndex < invoke.Method.Arguments.Count)
				values[memberIndex] = EvaluateExpression(invoke.Method.Arguments[argumentIndex++]);
			else
				values[memberIndex] = CreateDefaultValue(members[memberIndex].Type);
		Memory.Registers[invoke.Register] = new ValueInstance(targetType, values);
		return true;
	}

	private ValueInstance[] CreateConstructorValuesFromBinaryMembers(Type targetType, Invoke invoke,
		IReadOnlyList<BinaryMember> binaryMembers) =>
		CreateConstructorValuesFromBinaryMembers(targetType, invoke.Method.Arguments, binaryMembers);

	private ValueInstance[] CreateConstructorValuesFromBinaryMembers(Type targetType,
		IReadOnlyList<Expression> arguments, IReadOnlyList<BinaryMember> binaryMembers)
	{
		var values = new ValueInstance[binaryMembers.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count; memberIndex++)
		{
			var memberType = targetType.FindType(binaryMembers[memberIndex].FullTypeName) ??
				targetType.FindType(GetShortTypeName(binaryMembers[memberIndex].FullTypeName));
			if (memberType is { IsTrait: true })
				values[memberIndex] = CreateTraitInstance(memberType);
			else if (argumentIndex < arguments.Count)
				values[memberIndex] = EvaluateExpression(arguments[argumentIndex++]);
			else if (memberType != null)
				values[memberIndex] = CreateDefaultValue(memberType);
			else
				values[memberIndex] = new ValueInstance(executable.numberType, 0);
		}
		return values;
	}

	private static ValueInstance CreateDefaultValue(Type memberType) =>
		memberType.IsText
			? new ValueInstance("")
			: memberType.IsBoolean
				? new ValueInstance(memberType, false)
				: memberType.IsMutable
					// ReSharper disable once TailRecursiveCall
					? CreateDefaultValue(memberType.GetFirstImplementation())
					: new ValueInstance(memberType, 0);

	private static ValueInstance CreateTraitInstance(Type traitType)
	{
		var concreteType = traitType.FindType(traitType.Name is Type.TextWriter or Type.Logger
			? Type.System
			: traitType.Name);
		return concreteType != null
			? new ValueInstance(concreteType, Array.Empty<ValueInstance>())
			: new ValueInstance(traitType, 0);
	}

	/// <summary>
	/// Handles native trait method calls like logger.Log(...) by writing directly to Console.
	/// Logger delegates to TextWriter.Write which maps to System -> Console.WriteLine.
	/// </summary>
	private bool TryHandleNativeTraitMethod(Invoke invoke)
	{
		if (invoke.Method.Instance is not MemberCall memberCall)
			return false;
		var memberTypeName = memberCall.Member.Type.Name;
		if (memberTypeName is not (Type.Logger or Type.TextWriter or Type.System))
			return false;
		if (invoke.Method.Arguments.Count > 0)
		{
			var argValue = EvaluateExpression(invoke.Method.Arguments[0]);
			Console.WriteLine(argValue.ToExpressionCodeString());
		}
		return true;
	}

	/// <summary>
	/// Evaluates an arbitrary expression to a ValueInstance using the current VM state.
	/// Handles values, variables, member calls, binary operations, and method calls.
	/// </summary>
	private ValueInstance EvaluateExpression(Expression expression)
	{
		if (expression is Value value)
			return value.Data;
		if (expression is VariableCall variableCall && variableCall.IsConstant &&
			variableCall.Variable.InitialValue is Value constantValue)
			return constantValue.Data;
		if (expression is VariableCall or ParameterCall or Instance)
			return Memory.Frame.Get(expression.ToString());
		if (expression is MemberCall memberCall)
			return EvaluateMemberCall(memberCall);
		if (expression is Expressions.Binary binary)
			return EvaluateBinary(binary);
		if (expression is MethodCall methodCall)
			return EvaluateMethodCall(methodCall);
		return new ValueInstance(expression.ToString());
	}

	private ValueInstance EvaluateMemberCall(MemberCall memberCall)
	{
		if (memberCall.Instance != null)
		{
			var instanceValue = EvaluateExpression(memberCall.Instance);
			if (TryGetNativeLength(instanceValue, memberCall.Member.Name, out var lengthValue))
				return lengthValue;
			var typeInstance = instanceValue.TryGetValueTypeInstance();
			if (typeInstance != null && typeInstance.TryGetValue(memberCall.Member.Name, out var memberValue))
				return memberValue;
			if (instanceValue.IsText && memberCall.Member.Name is "characters" or "elements")
				return instanceValue;
		}
		if (Memory.Frame.TryGet(memberCall.ToString(), out var frameValue))
			return frameValue;
		if (Memory.Frame.TryGet(memberCall.Member.Name, out var memberFrameValue))
			return memberFrameValue;
		if (memberCall.Member.InitialValue is Value enumValue)
			return enumValue.Data;
		return new ValueInstance(memberCall.ToString());
	}

	private bool TryGetNativeLength(ValueInstance instance, string memberName, out ValueInstance result)
	{
		if (memberName is "Length" or "Count")
		{
			if (instance.IsText)
			{
				result = new ValueInstance(executable.numberType, instance.Text.Length);
				return true;
			}
			if (instance.IsList)
			{
				result = new ValueInstance(executable.numberType, instance.List.Items.Count);
				return true;
			}
		}
		result = default;
		return false;
	}

	private ValueInstance EvaluateBinary(Binary binary)
	{
		var left = EvaluateExpression(binary.Instance!);
		var right = EvaluateExpression(binary.Arguments[0]);
		return binary.Method.Name switch
		{
			BinaryOperator.Plus => AddValueInstances(left, right),
			BinaryOperator.Minus => SubtractValueInstances(left, right),
			BinaryOperator.Multiply => new ValueInstance(right.GetType(),
				left.Number * right.Number),
			BinaryOperator.Divide => new ValueInstance(right.GetType(),
				left.Number / right.Number),
			_ => new ValueInstance(left.GetType(), left.Number)
		};
	}

	private ValueInstance EvaluateMethodCall(MethodCall call)
	{
		if (call.Method.Name == Method.From)
			return EvaluateFromConstructor(call);
		if (call.Method.Name == BinaryOperator.To && call.Instance != null)
		{
			var rawValue = EvaluateExpression(call.Instance);
			if (call.ReturnType.IsText)
				return ConvertToText(rawValue);
			if (call.ReturnType.IsNumber)
				return rawValue.IsText
					? new ValueInstance(call.ReturnType, Convert.ToDouble(rawValue.Text))
					: rawValue;
		}
		var precompiledInstructions = GetPrecompiledMethodInstructions(call.Method);
		if (precompiledInstructions == null)
			throw new InvalidOperationException("No precompiled method instructions found for method call");
		var evaluatedArguments = call.Arguments.Select(EvaluateExpression).ToArray();
		var evaluatedInstance = call.Instance != null
			? EvaluateExpression(call.Instance)
			: (ValueInstance?)null;
		var precompiledResult = RunChildScope(precompiledInstructions,
			() => InitializeMethodCallScope(call, evaluatedArguments, evaluatedInstance));
		return precompiledResult ?? new ValueInstance(call.Method.ReturnType, 0);
	}

	private ValueInstance EvaluateFromConstructor(MethodCall call)
	{
		var targetType = call.ReturnType;
		var members = targetType.Members;
		if (members.Count == 0 && TryGetBinaryMembers(targetType, out var binaryMembers))
			return new ValueInstance(targetType,
				CreateConstructorValuesFromBinaryMembers(targetType, call.Arguments, binaryMembers));
		var values = new ValueInstance[members.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (members[memberIndex].Type.IsTrait)
				values[memberIndex] = CreateTraitInstance(members[memberIndex].Type);
			else if (argumentIndex < call.Arguments.Count)
				values[memberIndex] = EvaluateExpression(call.Arguments[argumentIndex++]);
			else
				values[memberIndex] = CreateDefaultValue(members[memberIndex].Type);
		return new ValueInstance(targetType, values);
	}

	private bool GetValueByKeyForDictionaryAndStoreInRegister(Invoke invoke)
	{
		if (invoke.Method.Method.Name != "Get" ||
			invoke.Method.Instance?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Type.Dictionary
			})
			return false;
		var keyArg = invoke.Method.Arguments[0];
		var keyData = keyArg is Value argValue
			? argValue.Data
			: Memory.Frame.Get(keyArg.ToString());
		var dictionary = Memory.Frame.Get(invoke.Method.Instance.ToString());
		var value = dictionary.GetDictionaryItems().
			FirstOrDefault(element => element.Key.Equals(keyData)).Value;
		if (!Equals(value, default(ValueInstance)))
			Memory.Registers[invoke.Register] = value;
		return true;
	}

	private void ExecuteReturn(ReturnInstruction returnInstruction)
	{
		Returns = Memory.Registers[returnInstruction.Register];
		instructionIndex = ExitExecutionLoopIndex;
	}

	private const int ExitExecutionLoopIndex = 100_000;
}