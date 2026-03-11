using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed class VirtualMachine(Package package)
{
	private readonly Type numberType = package.GetType(Type.Number);

	public VirtualMachine Execute(IList<Instruction> allInstructions)
	{
		Clear();
		foreach (var loopBegin in allInstructions.OfType<LoopBeginInstruction>())
			loopBegin.Reset();
		return RunInstructions(allInstructions);
	}

	private void Clear()
	{
		conditionFlag = false;
		instructionIndex = 0;
		instructions.Clear();
		Returns = null;
		Memory.Registers.Clear();
		Memory.Frame = new CallFrame();
	}

	private bool conditionFlag;
	private int instructionIndex;
	private IList<Instruction> instructions = new List<Instruction>();
	public ValueInstance? Returns { get; private set; }
	public Memory Memory { get; } = new();

	private VirtualMachine RunInstructions(IList<Instruction> allInstructions)
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
		TryInvokeInstruction(instruction);
		TryWriteToListInstruction(instruction);
		TryWriteToTableInstruction(instruction);
		TryRemoveInstruction(instruction);
		TryExecuteListCall(instruction);
		TryExecuteRest(instruction);
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
			Memory.Registers[writeToTableInstruction.Key], Memory.Registers[writeToTableInstruction.Value]);
	}

	private void TryLoopEndInstruction(Instruction instruction)
	{
		if (instruction is not LoopEndInstruction loopEnd)
			return;
		var loopBegin = loopEnd.Begin ?? FindLoopBeginFromSteps(loopEnd.Steps);
		loopBegin.LoopCount--;
		if (loopBegin.LoopCount <= 0)
			return;
		instructionIndex -= loopEnd.Steps + 1;
	}

	/// <summary>
	/// Fallback for manually constructed or deserialized LoopEndInstructions that don't have
	/// Begin set. Scans forward from the computed base position (at most 2 steps for range loops).
	/// </summary>
	private LoopBeginInstruction FindLoopBeginFromSteps(int steps)
	{
		var idx = instructionIndex - steps;
		while (idx < instructions.Count && instructions[idx] is not LoopBeginInstruction)
			idx++;
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
		var methodInstructions = GetByteCodeFromInvokedMethodCall(invoke);
		var result = RunChildScope(methodInstructions);
		if (result != null)
			Memory.Registers[invoke.Register] = result.Value;
	}

	/// <summary>
	/// Runs <paramref name="childInstructions"/> in a child <see cref="CallFrame"/> while reusing
	/// this interpreter instance. All mutable fields are saved on the call stack (zero heap
	/// allocations for the bookkeeping) and restored after the child finishes.
	/// </summary>
	private ValueInstance? RunChildScope(List<Instruction> childInstructions)
	{
		var savedInstructions = instructions;
		var savedIndex = instructionIndex;
		var savedConditionFlag = conditionFlag;
		var savedReturns = Returns;
		var savedFrame = Memory.Frame;
		Memory.Frame = new CallFrame(savedFrame);
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
			: Memory.Frame.TryGet(instanceExpr.ToString(), out var varValue)
				? varValue
				: throw new InvalidOperationException(); //ncrunch: no coverage
		var conversionType = invoke.Method.ReturnType;
		if (conversionType.IsText)
			Memory.Registers[invoke.Register] =
				rawValue.IsText
					? rawValue
					: new ValueInstance(rawValue.ToExpressionCodeString());
		else if (conversionType.IsNumber)
			Memory.Registers[invoke.Register] =
				rawValue.IsText
					? new ValueInstance(conversionType, Convert.ToDouble(rawValue.Text))
					: rawValue;
		return true;
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
		var values = new ValueInstance[members.Count];
		var argIndex = 0;
		for (var i = 0; i < members.Count; i++)
			if (members[i].Type.IsTrait)
				values[i] = CreateTraitInstance(members[i].Type); //ncrunch: no coverage
			else if (argIndex < invoke.Method.Arguments.Count)
				values[i] = EvaluateExpression(invoke.Method.Arguments[argIndex++]);
			else
				values[i] = new ValueInstance(members[i].Type, 0);
		Memory.Registers[invoke.Register] = new ValueInstance(targetType, values);
		return true;
	}

	private static ValueInstance CreateTraitInstance(Type traitType)
	{
		var concreteName = traitType.Name switch
		{
			Type.TextWriter => Type.System,
			Type.Logger => Type.System,
			_ => traitType.Name
		};
		var concreteType = traitType.FindType(concreteName);
		return concreteType != null
			? new ValueInstance(concreteType, System.Array.Empty<ValueInstance>())
			: new ValueInstance(traitType, 0);
	}

	/// <summary>
	/// Handles native trait method calls like logger.Log(...) by writing directly to Console.
	/// Logger delegates to TextWriter.Write which maps to System → Console.WriteLine.
	/// </summary>
	private bool TryHandleNativeTraitMethod(Invoke invoke)
	{
		if (invoke.Method?.Instance is not MemberCall memberCall)
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
		if (expression is VariableCall or ParameterCall)
			return Memory.Frame.Get(expression.ToString());
		if (expression is MemberCall memberCall)
			return EvaluateMemberCall(memberCall);
		if (expression is Binary binary)
			return EvaluateBinary(binary);
		if (expression is MethodCall methodCall)
			return EvaluateMethodCall(methodCall);
		return new ValueInstance(expression.ToString()); //ncrunch: no coverage
	}

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

	private ValueInstance EvaluateBinary(Binary binary)
	{
		var left = EvaluateExpression(binary.Instance!);
		var right = EvaluateExpression(binary.Arguments[0]);
		return binary.Method.Name switch
		{
			BinaryOperator.Plus => AddValueInstances(left, right),
			//ncrunch: no coverage start
			BinaryOperator.Minus => SubtractValueInstances(left, right),
			BinaryOperator.Multiply => new ValueInstance(right.GetType(),
				left.Number * right.Number),
			BinaryOperator.Divide => new ValueInstance(right.GetType(),
				left.Number / right.Number),
			_ => new ValueInstance(left.GetType(), left.Number)
		}; //ncrunch: no coverage end
	}

	private ValueInstance EvaluateMethodCall(MethodCall call)
	{
		if (call.Method.Name == Method.From)
			return EvaluateFromConstructor(call); //ncrunch: no coverage
		var instance = call.Instance != null
			? EvaluateExpression(call.Instance)
			: default;
		var args = call.Method.Parameters.Count > 0
			? call.Arguments.Select(EvaluateExpression).ToArray()
			: [];
		var argDict = new Dictionary<string, ValueInstance>();
		for (var i = 0; i < call.Method.Parameters.Count && i < args.Length; i++)
			argDict[call.Method.Parameters[i].Name] = args[i]; //ncrunch: no coverage
		var expressions = GetExpressionsFromMethod(call.Method);
		var invokedMethod = !instance.Equals(default(ValueInstance))
			? new InstanceInvokedMethod(expressions, argDict, instance, call.Method.ReturnType)
			: new InvokedMethod(expressions, argDict, call.Method.ReturnType);
		var childInstructions = new BytecodeGenerator(invokedMethod, new Registry()).Generate();
		var result = RunChildScope(childInstructions);
		return result ?? new ValueInstance(call.Method.ReturnType, 0);
	}

	//ncrunch: no coverage start
	private ValueInstance EvaluateFromConstructor(MethodCall call)
	{
		var targetType = call.ReturnType;
		var members = targetType.Members;
		var values = new ValueInstance[members.Count];
		var argIndex = 0;
		for (var i = 0; i < members.Count; i++)
			if (members[i].Type.IsTrait)
				values[i] = CreateTraitInstance(members[i].Type);
			else if (argIndex < call.Arguments.Count)
				values[i] = EvaluateExpression(call.Arguments[argIndex++]);
			else
				values[i] = new ValueInstance(members[i].Type, 0);
		return new ValueInstance(targetType, values);
	} //ncrunch: no coverage end

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

	private List<Instruction> GetByteCodeFromInvokedMethodCall(Invoke invoke)
	{
		if (invoke.Method?.Instance == null && invoke.Method?.Method != null &&
			invoke.PersistedRegistry != null)
			return new BytecodeGenerator(
				new InvokedMethod(GetExpressionsFromMethod(invoke.Method.Method),
					FormArgumentsForMethodCall(invoke), invoke.Method.Method.ReturnType),
				invoke.PersistedRegistry).Generate();
		if (!Memory.Frame.TryGet(invoke.Method?.Instance?.ToString() ??
			throw new InvalidOperationException(), out var instance))
			throw new VariableNotFoundInMemory(); //ncrunch: no coverage
		return new BytecodeGenerator(
			new InstanceInvokedMethod(GetExpressionsFromMethod(invoke.Method!.Method),
				FormArgumentsForMethodCall(invoke), instance, invoke.Method.Method.ReturnType),
			invoke.PersistedRegistry ?? throw new InvalidOperationException()).Generate();
	}

	private static IReadOnlyList<Expression> GetExpressionsFromMethod(Method method)
	{
		var result = method.GetBodyAndParseIfNeeded();
		return result is Body body
			? body.Expressions
			: [result];
	}

	private Dictionary<string, ValueInstance> FormArgumentsForMethodCall(Invoke invoke)
	{
		var arguments = new Dictionary<string, ValueInstance>();
		if (invoke.Method == null)
			return arguments; // ncrunch: no coverage
		for (var index = 0; index < invoke.Method.Method.Parameters.Count; index++)
		{
			if (index >= invoke.Method.Arguments.Count)
				break; //ncrunch: no coverage
			var argument = invoke.Method.Arguments[index];
			var argumentInstance = EvaluateExpression(argument);
			arguments.Add(invoke.Method.Method.Parameters[index].Name, argumentInstance);
		}
		return arguments;
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
		if (Memory.Frame.TryGet("index", out var indexValue))
			Memory.Frame.Set("index", new ValueInstance(numberType, indexValue.Number + 1));
		else
			Memory.Frame.Set("index", new ValueInstance(numberType, 0));
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
			Memory.Frame.Set("index", new ValueInstance(numberType, indexValue.Number + (isDecreasing
				? -1
				: 1)));
		else
			Memory.Frame.Set("index",
				new ValueInstance(numberType, loopBegin.StartIndexValue ?? 0));
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
		Memory.Frame.Set("value", new ValueInstance(numberType, index + 1));
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
			Memory.Registers[loadConstant.Register] = loadConstant.ValueInstance;
	}

	private void TryExecuteRest(Instruction instruction)
	{
		if (instruction is BinaryInstruction binary)
		{
			if (binary.IsConditional())
				TryConditionalOperationExecution(binary);
			else
				TryBinaryOperationExecution(binary);
		}
		else if (instruction is Jump jump)
			TryJumpOperation(jump);
		else if (instruction is JumpIf jumpIf)
			TryJumpIfOperation(jumpIf);
		else if (instruction is JumpToId jumpToId)
			TryJumpToIdOperation(jumpToId);
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

	private void TryJumpIfOperation(JumpIf instruction)
	{
		if (conditionFlag && instruction.InstructionType is InstructionType.JumpIfTrue ||
			!conditionFlag && instruction.InstructionType is InstructionType.JumpIfFalse ||
			instruction is JumpIfNotZero jumpIfNotZero &&
			Memory.Registers[jumpIfNotZero.Register].Number > 0)
			instructionIndex += Convert.ToInt32(instruction.Steps);
	}

	private void TryJumpToIdOperation(JumpToId instruction)
	{
		if (!conditionFlag && instruction.InstructionType is InstructionType.JumpToIdIfFalse ||
			conditionFlag && instruction.InstructionType is InstructionType.JumpToIdIfTrue)
		{
			var id = instruction.Id;
			var endIndex = instructions.IndexOf(instructions.First(jump =>
				jump.InstructionType is InstructionType.JumpEnd && jump is JumpToId jumpViaId &&
				jumpViaId.Id == id));
			if (endIndex != -1)
				instructionIndex = endIndex;
		}
	}

	public sealed class OperandsRequired : Exception;
	private sealed class VariableNotFoundInMemory : Exception;
}