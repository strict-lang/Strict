using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed class VirtualMachine(BinaryExecutable executable)
{
	public VirtualMachine(Package basePackage) : this(new BinaryExecutable(basePackage)) { }
	private readonly BinaryExecutable activeExecutable = executable;

	public VirtualMachine ExecuteRun(IReadOnlyDictionary<string, ValueInstance>? initialVariables = null) =>
		ExecuteExpression(activeExecutable.EntryPoint, initialVariables);

	public VirtualMachine ExecuteExpression(BinaryMethod method,
		IReadOnlyDictionary<string, ValueInstance>? initialVariables = null)
	{
		Clear(method.instructions, initialVariables);
		return RunInstructions(method.instructions);
	}

	private void Clear(IReadOnlyList<Instruction> allInstructions,
		IReadOnlyDictionary<string, ValueInstance>? initialVariables = null)
	{
		conditionFlag = false;
		instructionIndex = 0;
		instructions = [];
		Returns = null;
		Memory.Registers.Clear();
		Memory.Frame = new CallFrame();
		if (initialVariables != null)
			foreach (var (name, value) in initialVariables)
				Memory.Frame.Set(name, value);
		foreach (var loopBegin in allInstructions.OfType<LoopBeginInstruction>())
			loopBegin.Reset();
	}

	private bool conditionFlag;
	private int instructionIndex;
	private IReadOnlyList<Instruction> instructions = [];
	public ValueInstance? Returns { get; private set; }
	public Memory Memory { get; } = new();

	//TODO: this is still wrong, this are not allInstructions, just the entryPoint instructions, which should recursively call whatever is needed!
	private VirtualMachine RunInstructions(IReadOnlyList<Instruction> allInstructions)
	{
		instructions = allInstructions;
		for (instructionIndex = 0;
			instructionIndex is not -1 && instructionIndex < instructions.Count;
			instructionIndex++)
			ExecuteInstruction(instructions[instructionIndex]);
		return this;
	}

	private void ExecuteInstruction(Instruction instruction)
	{
		if (TryExecuteReturn(instruction))
			return;
		TryStoreInstructions(instruction);
		TryLoadInstructions(instruction);
		TryLoopInitInstruction(instruction);
		TryLoopEndInstruction(instruction);
		TryPrintInstruction(instruction);
		TryInvokeInstruction(instruction);
		TryWriteToListInstruction(instruction);
		TryWriteToTableInstruction(instruction);
		TryRemoveInstruction(instruction);
		TryExecuteListCall(instruction);
		TryExecuteRest(instruction);
	}

	private void TryPrintInstruction(Instruction instruction)
	{
		if (instruction is not PrintInstruction print)
			return;
		if (print.ValueRegister.HasValue)
			Console.WriteLine(print.TextPrefix + Memory.Registers[print.ValueRegister.Value].ToExpressionCodeString());
		else
			Console.WriteLine(print.TextPrefix); //ncrunch: no coverage
	}

	private void TryRemoveInstruction(Instruction instruction)
	{
		if (instruction is not RemoveInstruction removeInstruction)
			return;
		var item = Memory.Registers[removeInstruction.Register];
		Memory.Frame.Get(removeInstruction.Identifier).List.Items.RemoveAll(existingItem => existingItem.Equals(item));
	}

	private void TryExecuteListCall(Instruction instruction)
	{
		if (instruction is not ListCallInstruction listCallInstruction)
			return;
		var indexValue = (int)Memory.Registers[listCallInstruction.IndexValueRegister].Number;
		var variableListElement = Memory.Frame.Get(listCallInstruction.Identifier).List.Items[indexValue];
		Memory.Registers[listCallInstruction.Register] = variableListElement;
	}

	private void TryWriteToListInstruction(Instruction instruction)
	{
		if (instruction is not WriteToListInstruction writeToListInstruction)
			return;
		Memory.AddToCollectionVariable(writeToListInstruction.Identifier,
			Memory.Registers[writeToListInstruction.Register]);
	}

	private void TryWriteToTableInstruction(Instruction instruction)
	{
		if (instruction is not WriteToTableInstruction writeToTableInstruction)
			return;
		Memory.AddToDictionary(writeToTableInstruction.Identifier,
			Memory.Registers[writeToTableInstruction.Register], Memory.Registers[writeToTableInstruction.Value]);
	}

	private void TryLoopEndInstruction(Instruction instruction)
	{
		if (instruction is not LoopEndInstruction loopEnd)
			return;
		var loopBegin = loopEnd.Begin ?? FindLoopBeginByScanning(loopEnd.Steps);
		loopBegin.LoopCount--;
		if (loopBegin.LoopCount <= 0)
			return;
		instructionIndex = GetInstructionIndex(loopBegin) - 1;
	}

	private int GetInstructionIndex(Instruction instruction)
	{
		for (var index = 0; index < instructions.Count; index++)
			if (ReferenceEquals(instructions[index], instruction))
				return index;
		return -1;
	}

	/// <summary>
	/// Fallback for deserialized LoopEndInstructions that don't have Begin set.
	/// Uses Steps as a hint to find the LoopBeginInstruction by scanning.
	/// </summary>
	private LoopBeginInstruction FindLoopBeginByScanning(int steps)
	{
		var idx = Math.Max(0, instructionIndex - steps);
		while (idx < instructions.Count && instructions[idx] is not LoopBeginInstruction)
			idx++; //ncrunch: no coverage
		return idx < instructions.Count
			? (LoopBeginInstruction)instructions[idx]
			: throw new InvalidOperationException("No matching LoopBeginInstruction found for LoopEnd");
	}

	private void TryInvokeInstruction(Instruction instruction)
	{
		if (instruction is not Invoke { Method: not null } invoke ||
			TryCreateEmptyDictionaryInstance(invoke) || TryHandleFromConstructor(invoke) ||
			TryHandleNativeTraitMethod(invoke) || TryHandleToConversion(invoke) ||
			TryHandleIncrementDecrement(invoke) || GetValueByKeyForDictionaryAndStoreInRegister(invoke))
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

	private List<Instruction>? GetPrecompiledMethodInstructions(Method method)
	{
    var foundInstructions = activeExecutable.FindInstructions(method.Type, method) ??
			activeExecutable.FindInstructions(method.Type.Name, method.Name, method.Parameters.Count,
				method.ReturnType.Name) ??
			activeExecutable.FindInstructions(nameof(Strict) + Context.ParentSeparator + method.Type.Name,
				method.Name, method.Parameters.Count, method.ReturnType.Name);
		return foundInstructions == null
			? null
			//TODO: find all [.. with existing list and no changes, all those cases need to be removed, there is a crazy amount of those added (54 wtf)!
			: [.. foundInstructions];
	}

	private List<Instruction>? GetPrecompiledMethodInstructions(Invoke invoke) =>
		invoke.Method == null
			? null
			: GetPrecompiledMethodInstructions(invoke.Method.Method);

	private void InitializeMethodCallScope(MethodCall methodCall,
		IReadOnlyList<ValueInstance>? evaluatedArguments = null,
		ValueInstance? evaluatedInstance = null)
	{
		for (var parameterIndex = 0; parameterIndex < methodCall.Method.Parameters.Count &&
			//ncrunch: no coverage start
			parameterIndex < methodCall.Arguments.Count; parameterIndex++)
			Memory.Frame.Set(methodCall.Method.Parameters[parameterIndex].Name,
				evaluatedArguments != null
					? evaluatedArguments[parameterIndex]
					: EvaluateExpression(methodCall.Arguments[parameterIndex]));
		if (methodCall.Instance == null)
			return;
		//ncrunch: no coverage end
		var instance = evaluatedInstance ?? EvaluateExpression(methodCall.Instance);
		Memory.Frame.Set(Type.ValueLowercase, instance, isMember: true);
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null)
		{
			if (TrySetScopeMembersFromTypeMembers(typeInstance) ||
				TrySetScopeMembersFromBinaryMembers(typeInstance))
				return;
		}
		//ncrunch: no coverage start
		var firstNonTraitMember = instance.GetType().Members.FirstOrDefault(member =>
			!member.Type.IsTrait);
		if (firstNonTraitMember != null)
			Memory.Frame.Set(firstNonTraitMember.Name, instance, isMember: true);
		//ncrunch: no coverage end
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
		foreach (var (typeName, typeData) in activeExecutable.MethodsPerType)
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

	private ValueInstance? RunChildScope(List<Instruction> childInstructions,
		Action? initializeScope = null)
	{
		var savedInstructions = instructions;
		var savedIndex = instructionIndex;
		var savedConditionFlag = conditionFlag;
		var savedReturns = Returns;
		var savedFrame = Memory.Frame;
		Memory.Frame = new CallFrame(savedFrame);
		initializeScope?.Invoke();
		Returns = null;
		RunInstructions(childInstructions);
		var result = Returns;
		Memory.Frame.Clear();
		Memory.Frame = savedFrame;
		instructions = savedInstructions;
		instructionIndex = savedIndex;
		conditionFlag = savedConditionFlag;
		Returns = savedReturns;
		return result;
	}

	private bool TryHandleIncrementDecrement(Invoke invoke)
	{
		var methodName = invoke.Method?.Method.Name;
		if (methodName != "Increment" && methodName != "Decrement")
			return false;
		if (invoke.Method!.Instance == null ||
			!Memory.Frame.TryGet(invoke.Method.Instance.ToString(), out var current))
			return false; //ncrunch: no coverage
		var delta = methodName == "Increment"
			? 1.0
			: -1.0;
		Memory.Registers[invoke.Register] =
			new ValueInstance(current.GetType(), current.Number + delta);
		return true;
	}

	private bool TryHandleToConversion(Invoke invoke)
	{
		if (invoke.Method?.Method.Name != BinaryOperator.To)
			return false;
		var instanceExpr = invoke.Method.Instance ?? throw new InvalidOperationException();
		var rawValue = instanceExpr is Value constValue
			? constValue.Data
			: Memory.Frame.TryGet(instanceExpr.ToString(), out var variableValue)
				? variableValue
				: throw new InvalidOperationException(); //ncrunch: no coverage
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
			return rawValue; //ncrunch: no coverage
		if (rawValue.TryGetValueTypeInstance() is { } typeInstance)
		{ //ncrunch: no coverage start
			var members = typeInstance.ReturnType.Members;
			var memberValues = new List<string>(typeInstance.Values.Length);
			for (var memberIndex = 0; memberIndex < typeInstance.Values.Length && memberIndex < members.Count; memberIndex++)
				if (!members[memberIndex].Type.IsTrait && members[memberIndex].Type.Name is not
					(Type.Logger or Type.TextWriter or Type.System))
					memberValues.Add(typeInstance.Values[memberIndex].ToExpressionCodeString());
			return memberValues.Count == 0
				? new ValueInstance(typeInstance.ReturnType.Name)
				: new ValueInstance("(" + string.Join(", ", memberValues) + ")");
		} //ncrunch: no coverage end
		return new ValueInstance(rawValue.ToExpressionCodeString());
	}

	private bool TryCreateEmptyDictionaryInstance(Invoke invoke)
	{
		if (invoke.Method?.Instance != null || invoke.Method?.Method.Name != Method.From ||
			invoke.Method?.ReturnType is not GenericTypeImplementation
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
		if (invoke.Method?.Method.Name != Method.From || invoke.Method.Instance != null)
			return false;
		var targetType = invoke.Method.ReturnType;
		if (targetType is GenericTypeImplementation)
			return false; //ncrunch: no coverage
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
				values[memberIndex] = new ValueInstance(members[memberIndex].Type, 0);
		Memory.Registers[invoke.Register] = new ValueInstance(targetType, values);
		return true;
	}

	private ValueInstance[] CreateConstructorValuesFromBinaryMembers(Type targetType, Invoke invoke,
		IReadOnlyList<BinaryMember> binaryMembers)
	{
		var values = new ValueInstance[binaryMembers.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count; memberIndex++)
		{
			var memberType = targetType.FindType(binaryMembers[memberIndex].FullTypeName) ??
				targetType.FindType(GetShortTypeName(binaryMembers[memberIndex].FullTypeName));
			if (memberType != null && memberType.IsTrait)
				values[memberIndex] = CreateTraitInstance(memberType);
			else if (argumentIndex < invoke.Method.Arguments.Count)
				values[memberIndex] = EvaluateExpression(invoke.Method.Arguments[argumentIndex++]);
			else
				values[memberIndex] = new ValueInstance(activeExecutable.numberType, 0);
		}
		return values;
	}

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
		if (invoke.Method?.Instance is not MemberCall memberCall)
			return false;
		var memberTypeName = memberCall.Member.Type.Name;
		if (memberTypeName is not (Type.Logger or Type.TextWriter or Type.System))
			return false;
		//ncrunch: no coverage start
		if (invoke.Method.Arguments.Count > 0)
		{
			var argValue = EvaluateExpression(invoke.Method.Arguments[0]);
			Console.WriteLine(argValue.ToExpressionCodeString());
		}
		return true;
	} //ncrunch: no coverage end

	/// <summary>
	/// Evaluates an arbitrary expression to a ValueInstance using the current VM state.
	/// Handles values, variables, member calls, binary operations, and method calls.
	/// </summary>
	private ValueInstance EvaluateExpression(Expression expression)
	{
		if (expression is Value value)
			return value.Data;
		if (expression is VariableCall or ParameterCall or Instance)
			return Memory.Frame.Get(expression.ToString());
		if (expression is MemberCall memberCall)
			return EvaluateMemberCall(memberCall);
		//ncrunch: no coverage start
		if (expression is Expressions.Binary binary)
			return EvaluateBinary(binary);
		if (expression is MethodCall methodCall)
			return EvaluateMethodCall(methodCall);
		return new ValueInstance(expression.ToString());
	} //ncrunch: no coverage end

	private ValueInstance EvaluateMemberCall(MemberCall memberCall)
	{
		if (memberCall.Instance != null &&
			Memory.Frame.TryGet(memberCall.Instance.ToString(), out var instanceValue))
		{ //ncrunch: no coverage start
			var typeInstance = instanceValue.TryGetValueTypeInstance();
			if (typeInstance != null && typeInstance.TryGetValue(memberCall.Member.Name, out var memberValue))
				return memberValue;
		} //ncrunch: no coverage end
		if (Memory.Frame.TryGet(memberCall.ToString(), out var frameValue))
			return frameValue;
		//ncrunch: no coverage start
		if (memberCall.Member.InitialValue is Value enumValue)
			return enumValue.Data;
		return new ValueInstance(memberCall.ToString());
	} //ncrunch: no coverage end

	//ncrunch: no coverage start
	private ValueInstance EvaluateBinary(Expressions.Binary binary)
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
	} //ncrunch: no coverage end

	//ncrunch: no coverage start
	private ValueInstance EvaluateMethodCall(MethodCall call)
	{
		if (call.Method.Name == Method.From)
			return EvaluateFromConstructor(call); //ncrunch: no coverage
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
	} //ncrunch: no coverage end

	private ValueInstance EvaluateFromConstructor(MethodCall call)
	{
		var targetType = call.ReturnType;
		var members = targetType.Members;
		var values = new ValueInstance[members.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (members[memberIndex].Type.IsTrait)
				values[memberIndex] = CreateTraitInstance(members[memberIndex].Type);
			else if (argumentIndex < call.Arguments.Count)
				values[memberIndex] = EvaluateExpression(call.Arguments[argumentIndex++]);
			else
				values[memberIndex] = new ValueInstance(members[memberIndex].Type, 0);
		return new ValueInstance(targetType, values);
	}

	private bool GetValueByKeyForDictionaryAndStoreInRegister(Invoke invoke)
	{
		if (invoke.Method?.Method.Name != "Get" ||
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

	private bool TryExecuteReturn(Instruction instruction)
	{
		if (instruction is not ReturnInstruction returnInstruction)
			return false;
		Returns = Memory.Registers[returnInstruction.Register];
		instructionIndex = -2;
		return true;
	}

	private void TryLoopInitInstruction(Instruction instruction)
	{
		if (instruction is not LoopBeginInstruction loopBegin)
			return;
		if (loopBegin.IsRange)
			ProcessRangeLoopIteration(loopBegin);
		else
			ProcessCollectionLoopIteration(loopBegin);
	}

	private void ProcessCollectionLoopIteration(LoopBeginInstruction loopBegin)
	{
		if (!Memory.Registers.TryGet(loopBegin.Register, out var iterableVariable))
			return; //ncrunch: no coverage
		Memory.Frame.Set("index", Memory.Frame.TryGet("index", out var indexValue)
     ? new ValueInstance(activeExecutable.numberType, indexValue.Number + 1)
			: new ValueInstance(activeExecutable.numberType, 0));
		if (!loopBegin.IsInitialized)
		{
			loopBegin.LoopCount = GetLength(iterableVariable);
			loopBegin.IsInitialized = true;
		}
		AlterValueVariable(iterableVariable, loopBegin);
		if (loopBegin.LoopCount <= 0)
		{
			var skipTo = instructionIndex + 1;
			while (skipTo < instructions.Count && instructions[skipTo] is not LoopEndInstruction)
				skipTo++;
			instructionIndex = skipTo;
		}
	}

	private void ProcessRangeLoopIteration(LoopBeginInstruction loopBegin)
	{
		if (!loopBegin.IsInitialized)
		{
			var startIndex = Convert.ToInt32(Memory.Registers[loopBegin.Register].Number);
			var endIndex = Convert.ToInt32(Memory.Registers[loopBegin.EndIndex!.Value].Number);
			loopBegin.InitializeRangeState(startIndex, endIndex);
		}
		var isDecreasing = loopBegin.IsDecreasing ?? false;
		if (Memory.Frame.TryGet("index", out var indexValue))
      Memory.Frame.Set("index", new ValueInstance(activeExecutable.numberType, indexValue.Number +
				(isDecreasing ? -1 : 1)));
		else
      Memory.Frame.Set("index", new ValueInstance(activeExecutable.numberType,
				loopBegin.StartIndexValue ?? 0));
		Memory.Frame.Set("value", Memory.Frame.Get("index"));
	}

	private static int GetLength(ValueInstance iterableInstance)
	{
		if (iterableInstance.IsText)
			return iterableInstance.Text.Length;
		if (iterableInstance.IsList)
			return iterableInstance.List.Items.Count;
		return (int)iterableInstance.Number;
	}

	private void AlterValueVariable(ValueInstance iterableVariable, LoopBeginInstruction loopBegin)
	{
		var index = (int)Memory.Frame.Get("index").Number;
		if (iterableVariable.IsText)
		{
			if (index < iterableVariable.Text.Length)
				Memory.Frame.Set("value", new ValueInstance(iterableVariable.Text[index].ToString()));
			return;
		}
		if (iterableVariable.IsList)
		{
			var items = iterableVariable.List.Items;
			if (index < items.Count)
				Memory.Frame.Set("value", items[index]);
			else
				loopBegin.LoopCount = 0;
			return;
		}
   Memory.Frame.Set("value", new ValueInstance(activeExecutable.numberType, index + 1));
	}

	private void TryStoreInstructions(Instruction instruction)
	{
		if (instruction.InstructionType > InstructionType.StoreSeparator)
			return;
		if (instruction is SetInstruction set)
			Memory.Registers[set.Register] = set.ValueInstance;
		else if (instruction is StoreVariableInstruction storeVariable)
		{
			var value = storeVariable.ValueInstance;
			// Create defensive copy to isolate list state between separate Execute() calls when lists are mutated in-place
			if (value.IsList)
				value = new ValueInstance(value.List.ReturnType, value.List.Items.ToArray());
			Memory.Frame.Set(storeVariable.Identifier, value, storeVariable.IsMember);
		}
		else if (instruction is StoreFromRegisterInstruction storeFromRegister)
			Memory.Frame.Set(storeFromRegister.Identifier, Memory.Registers[storeFromRegister.Register]);
	}

	private void TryLoadInstructions(Instruction instruction)
	{
		if (instruction is LoadVariableToRegister loadVariable)
			Memory.Registers[loadVariable.Register] =
				Memory.Frame.Get(loadVariable.Identifier);
		else if (instruction is LoadConstantInstruction loadConstant)
			Memory.Registers[loadConstant.Register] = loadConstant.Constant;
	}

	private void TryExecuteRest(Instruction instruction)
	{
    switch (instruction)
		{
		case BinaryInstruction binary:
			if (binary.IsConditional())
				TryConditionalOperationExecution(binary);
			else
				TryBinaryOperationExecution(binary);
			break;
		case JumpIfNotZero jumpIfNotZero:
			TryJumpIfOperation(jumpIfNotZero);
			break;
		case Jump jump:
			TryJumpOperation(jump);
			break;
		case JumpToId jumpToId:
			TryJumpToIdOperation(jumpToId);
			break;
		}
	}

	private void TryBinaryOperationExecution(BinaryInstruction instruction)
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
			_ => Memory.Registers[instruction.Registers[^1]] //ncrunch: no coverage
		};
	}

	private static ValueInstance AddValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
		{
			// Mutates left's list in-place; caller's defensive copy in TryStoreInstructions ensures isolation
			left.List.Items.Add(right);
			return left;
		}
		if (left.IsText || right.IsText)
			return new ValueInstance((left.IsText
				? left.Text
				: left.Number.ToString()) + (right.IsText
				? right.Text
				: right.Number.ToString()));
		return new ValueInstance(right.GetType(), left.Number + right.Number);
	}

	private static ValueInstance SubtractValueInstances(ValueInstance left, ValueInstance right)
	{
		if (!left.IsList)
			return new ValueInstance(left.GetType(), left.Number - right.Number);
		var items = new List<ValueInstance>(left.List.Items);
		var removeIndex = items.FindIndex(item => item.Equals(right));
		if (removeIndex >= 0)
			items.RemoveAt(removeIndex);
		return new ValueInstance(left.List.ReturnType, items.ToArray());
	}

	private (ValueInstance, ValueInstance) GetOperands(BinaryInstruction instruction) =>
		instruction.Registers.Length < 2
			? throw new OperandsRequired()
			: (Memory.Registers[instruction.Registers[1]], Memory.Registers[instruction.Registers[0]]);

	private void TryConditionalOperationExecution(BinaryInstruction instruction)
	{
		var (right, left) = GetOperands(instruction);
		conditionFlag = instruction.InstructionType switch
		{
			InstructionType.GreaterThan => left.Number > right.Number,
			InstructionType.LessThan => left.Number < right.Number,
			InstructionType.Equal => left.Equals(right),
			InstructionType.NotEqual => !left.Equals(right),
			_ => false //ncrunch: no coverage
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
			if (instructions[index] is JumpToId { InstructionType: InstructionType.JumpEnd } jumpEnd &&
				jumpEnd.Id == id)
				return index;
		return -1;
	}

	public sealed class OperandsRequired : Exception;
	private sealed class VariableNotFoundInMemory : Exception;
}