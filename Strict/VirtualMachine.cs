using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

//ncrunch: no coverage start, performance is very bad when NCrunch is tracking every line
public sealed class VirtualMachine(BinaryExecutable executable)
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
		return RunInstructions(method.instructions);
	}

	private bool conditionFlag;
	private int instructionIndex;
	//TODO: find all IReadOnlyList here and remove, also why do we copy so many lists around, use BinaryMethod!
	private IReadOnlyList<Instruction> instructions = [];
	public ValueInstance? Returns { get; private set; }
	public Memory Memory { get; } = new();
	private const int MaxCallDepth = 64;
	private readonly ValueInstance[][] registerStack = new ValueInstance[MaxCallDepth][];
	private int registerStackDepth;
	private readonly CallFrame[] framePool = new CallFrame[MaxCallDepth];
	private int framePoolDepth;

	private VirtualMachine RunInstructions(IReadOnlyList<Instruction> blockInstructions)
	{
		foreach (var loopBegin in blockInstructions.OfType<LoopBeginInstruction>())
			loopBegin.Reset();
		instructions = blockInstructions;
		var instructionsLength = instructions.Count;
		for (instructionIndex = 0; instructionIndex < instructionsLength; instructionIndex++)
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
			Console.WriteLine(print.TextPrefix);
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
		var argCount = invoke.Method.Arguments.Count;
		var evaluatedArgs = argCount == 0
			? Array.Empty<ValueInstance>()
			: new ValueInstance[argCount];
		for (var argIndex = 0; argIndex < argCount; argIndex++)
			evaluatedArgs[argIndex] = EvaluateExpression(invoke.Method.Arguments[argIndex]);
		var evaluatedInstance = invoke.Method.Instance != null
			? EvaluateExpression(invoke.Method.Instance)
			: (ValueInstance?)null;
		var invokeInstructions = invoke.CachedInstructions ??= GetPrecompiledMethodInstructions(invoke) ??
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
			method.Parameters.Count, method.ReturnType.Name) ??
		FindInstructionsWithStrippedPackagePrefix(method);

	private IReadOnlyList<Instruction>? FindInstructionsWithStrippedPackagePrefix(Method method)
	{
		var fullName = method.Type.FullName;
		var strictPrefix = nameof(Strict) + Context.ParentSeparator;
		return fullName.StartsWith(strictPrefix, StringComparison.Ordinal)
			? executable.FindInstructions(fullName[strictPrefix.Length..], method.Name,
				method.Parameters.Count, method.ReturnType.Name)
			: null;
	}

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
		var depth = registerStackDepth++;
		if (registerStack[depth] == null)
			registerStack[depth] = new ValueInstance[16];
		Memory.Registers.SaveTo(registerStack[depth]);
		var frame = framePoolDepth > 0
			? framePool[--framePoolDepth]
			: new CallFrame();
		frame.Reset(savedFrame);
		Memory.Frame = frame;
		initializeScope?.Invoke();
		Returns = null;
		RunInstructions(childInstructions);
		var result = Returns;
		frame.Reset(null);
		if (framePoolDepth < MaxCallDepth)
			framePool[framePoolDepth++] = frame;
		Memory.Frame = savedFrame;
		Memory.Registers.RestoreFrom(registerStack[--registerStackDepth]);
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
			!Memory.Frame.TryGet(GetFrameKey(invoke.Method.Instance), out var current))
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
		var conversionType = invoke.Method.ReturnType;
		var rawValue = EvaluateExpression(invoke.Method.Instance ?? throw new InvalidOperationException());
		if (conversionType.IsText && !invoke.Method.Method.IsTrait &&
			invoke.Method.Method.Type == rawValue.GetType() &&
			rawValue.TryGetValueTypeInstance() != null)
			return false;
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
			return new ValueInstance(typeInstance.ToAutomaticText());
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
		var hasBinaryMembers = TryGetBinaryMembers(targetType, out var binaryMembers);
		if (members.Count == 0 && hasBinaryMembers)
		{
			Memory.Registers[invoke.Register] = new ValueInstance(targetType,
				CreateConstructorValuesFromBinaryMembers(targetType, invoke, binaryMembers));
			return true;
		}
		var values = new ValueInstance[members.Count];
		for (var parameterIndex = 0; parameterIndex < invoke.Method.Method.Parameters.Count; parameterIndex++)
		{
			var parameter = invoke.Method.Method.Parameters[parameterIndex];
			var memberIndex = FindMemberIndex(members, parameter.Name);
			if (memberIndex == -1)
				continue;
			var memberInitialValue = members[memberIndex].InitialValue;
			values[memberIndex] = parameterIndex < invoke.Method.Arguments.Count
				? EvaluateExpression(invoke.Method.Arguments[parameterIndex])
				: parameter.DefaultValue != null
					? EvaluateExpression(parameter.DefaultValue)
					: memberInitialValue != null
						? EvaluateExpression(memberInitialValue)
						: hasBinaryMembers && TryGetBinaryMemberInitialValue(binaryMembers, memberIndex,
							out var initialValue)
							? initialValue
							: CreateDefaultValue(members[memberIndex].Type);
		}
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (!values[memberIndex].HasValue)
			{
				var memberInitialValue = members[memberIndex].InitialValue;
				values[memberIndex] = members[memberIndex].Type.IsTrait
					? CreateTraitInstance(members[memberIndex].Type)
					: memberInitialValue != null
						? EvaluateExpression(memberInitialValue)
						: hasBinaryMembers && TryGetBinaryMemberInitialValue(binaryMembers, memberIndex,
							out var initialValue)
							? initialValue
							: CreateDefaultValue(members[memberIndex].Type);
			}
		TryPreFillConstrainedListMembers(targetType, values);
		Memory.Registers[invoke.Register] = new ValueInstance(targetType, values);
		return true;
	}

	private static bool TryGetBinaryMemberInitialValue(IReadOnlyList<BinaryMember> binaryMembers,
		int memberIndex, out ValueInstance value)
	{
		if (memberIndex < binaryMembers.Count &&
			binaryMembers[memberIndex].InitialValueExpression is SetInstruction setInstruction)
		{
			value = setInstruction.ValueInstance;
			return true;
		}
		value = default;
		return false;
	}

	private static int FindMemberIndex(IReadOnlyList<Member> members, string name)
	{
		for (var index = 0; index < members.Count; index++)
			if (members[index].Name.Equals(name, StringComparison.OrdinalIgnoreCase))
				return index;
		return -1;
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

	private void TryPreFillConstrainedListMembers(Type targetType, ValueInstance[] values)
	{
		var members = targetType.Members;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
		{
			if (!values[memberIndex].IsList || values[memberIndex].List.Items.Count > 0 ||
				members[memberIndex].Constraints == null)
				continue;
			var length = TryGetConstrainedLength(targetType, values, members[memberIndex]);
			if (length is not > 0)
				continue;
			var elementType = members[memberIndex].Type is GenericTypeImplementation genericList
				? genericList.ImplementationTypes[0]
				: members[memberIndex].Type;
			if (elementType.Members.FirstOrDefault(member => member.Name == Type.ElementsLowercase)?.Type is
				GenericTypeImplementation nestedElementsList)
				elementType = nestedElementsList.ImplementationTypes[0];
			var elements = new ValueInstance[length.Value];
			var defaultElement = CreateDefaultComplexValue(elementType);
			Array.Fill(elements, defaultElement);
			values[memberIndex] = new ValueInstance(members[memberIndex].Type, elements);
		}
	}

	private static ValueInstance CreateDefaultValue(Type memberType) =>
	(memberType.IsMutable
		? memberType.GetFirstImplementation()
		: memberType).IsList
		? new ValueInstance(memberType, Array.Empty<ValueInstance>())
		: (memberType.IsMutable
			? memberType.GetFirstImplementation()
			: memberType).IsDictionary
			? new ValueInstance(memberType, new Dictionary<ValueInstance, ValueInstance>())
			: memberType.IsText
				? new ValueInstance("")
				: memberType.IsBoolean
					? new ValueInstance(memberType, false)
					: memberType.IsMutable
						// ReSharper disable once TailRecursiveCall
						? CreateDefaultValue(memberType.GetFirstImplementation())
						: new ValueInstance(memberType, 0);

	private static ValueInstance CreateDefaultComplexValue(Type type)
	{
		if (type.IsList || type.IsDictionary || type.IsText || type.IsBoolean || type.IsNumber ||
			type.IsNone)
			return CreateDefaultValue(type);
		var members = type.Members;
		if (members.Count == 0)
			return CreateDefaultValue(type);
		var values = new ValueInstance[members.Count];
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			values[memberIndex] = members[memberIndex].Type.IsTrait
				? CreateTraitInstance(members[memberIndex].Type)
				: members[memberIndex].InitialValue is Value initialValue
					? initialValue.Data
					: CreateDefaultValue(members[memberIndex].Type);
		return new ValueInstance(type, values);
	}

	private int? TryGetConstrainedLength(Type targetType, ValueInstance[] values, Member member)
	{
		foreach (var constraint in member.Constraints!)
		{
			if (constraint is not Expressions.Binary { Method.Name: BinaryOperator.Is } binary ||
				binary.Instance?.ToString() != "Length")
				continue;
			var rhs = binary.Arguments[0];
			if (rhs is Value numberValue)
				return (int)numberValue.Data.Number;
			return TryEvaluateLengthInMemberScope(targetType, values, rhs) ??
				TryResolveMemberMethodLength(targetType, values, rhs);
		}
		return null;
	}

	private int? TryResolveMemberMethodLength(Type targetType, ValueInstance[] values,
		Expression rhs)
	{
		var rhsText = rhs.ToString();
		var dotIndex = rhsText.IndexOf('.');
		if (dotIndex <= 0)
			return null;
		var memberName = rhsText[..dotIndex];
		var methodName = rhsText[(dotIndex + 1)..];
		for (var memberIndex = 0; memberIndex < targetType.Members.Count; memberIndex++)
		{
			if (!targetType.Members[memberIndex].Name.Equals(memberName,
					StringComparison.OrdinalIgnoreCase) || !values[memberIndex].HasValue)
				continue;
			var memberValue = values[memberIndex];
			var typeInstance = memberValue.TryGetValueTypeInstance();
			var method = typeInstance?.ReturnType.FindMethod(methodName, []);
			if (method == null)
				continue;
			var methodInstructions = GetPrecompiledMethodInstructions(method);
			if (methodInstructions != null)
			{
				var result = RunChildScope(methodInstructions, () =>
				{
					Memory.Frame.Set(Type.ValueLowercase, memberValue, isMember: true);
					TrySetScopeMembersFromTypeMembers(typeInstance!);
				});
				if (result.HasValue)
					return (int)result.Value.Number;
			}
			else
			{
				var bodyResult = TryEvaluateMethodBodyWithInstance(method, memberValue, typeInstance!);
				if (bodyResult != null)
					return bodyResult;
			}
		}
		return null;
	}

	private int? TryEvaluateMethodBodyWithInstance(Method method, ValueInstance memberValue,
		ValueTypeInstance typeInstance)
	{
		try
		{
			var body = method.GetBodyAndParseIfNeeded();
			if (body is not Body { Expressions.Count: > 0 } methodBody)
				return null;
			var savedFrame = Memory.Frame;
			Memory.Frame = new CallFrame();
			Memory.Frame.Set(Type.ValueLowercase, memberValue, isMember: true);
			TrySetScopeMembersFromTypeMembers(typeInstance);
			try
			{
				var lastExpression = methodBody.Expressions[^1];
				return (int)EvaluateExpression(lastExpression).Number;
			}
			finally
			{
				Memory.Frame = savedFrame;
			}
		}
		catch
		{
			return null;
		}
	}

	private int? TryEvaluateLengthInMemberScope(Type targetType, ValueInstance[] values,
		Expression lengthExpression)
	{
		var savedFrame = Memory.Frame;
		Memory.Frame = new CallFrame();
		try
		{
			for (var memberIndex = 0; memberIndex < targetType.Members.Count; memberIndex++)
				if (values[memberIndex].HasValue)
					Memory.Frame.Set(targetType.Members[memberIndex].Name, values[memberIndex],
						isMember: true);
			return (int)EvaluateExpression(lengthExpression).Number;
		}
		catch
		{
			return null;
		}
		finally
		{
			Memory.Frame = savedFrame;
		}
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
      return Memory.Frame.Get(GetFrameKey(expression));
		if (expression is MemberCall memberCall)
			return EvaluateMemberCall(memberCall);
		if (expression is Expressions.Binary binary)
			return EvaluateBinary(binary);
		if (expression is MethodCall methodCall)
			return EvaluateMethodCall(methodCall);
		if (expression is ListCall listCall)
			return EvaluateListCallExpression(listCall);
    var frameKey = GetFrameKey(expression);
		return Memory.Frame.TryGet(frameKey, out var frameValue)
			? frameValue
     : new ValueInstance(frameKey);
	}

	private static string GetFrameKey(Expression expression) =>
		expression switch
		{
			VariableCall variableCall => variableCall.Variable.Name,
			ParameterCall parameterCall => parameterCall.Parameter.Name,
			Instance => Type.ValueLowercase,
			MemberCall memberCall => GetMemberCallFrameKey(memberCall),
			_ => expression.ToString()
		};

	private static string GetMemberCallFrameKey(MemberCall memberCall)
	{
		if (memberCall.Instance == null)
			return memberCall.ToString();
		var instanceKey = GetFrameKey(memberCall.Instance);
		return string.Concat(instanceKey, ".", memberCall.Member.Name);
	}

	private ValueInstance EvaluateListCallExpression(ListCall listCall)
	{
		var listValue = EvaluateExpression(listCall.List);
		var indexValue = EvaluateExpression(listCall.Index);
		var index = (int)indexValue.Number;
		if (listValue.IsList || listValue.IsText || listValue.TryGetValueTypeInstance()?.ReturnType.IsList == true)
			return listValue.GetIteratorValue(executable.characterType, index);
		if (listValue.TryGetValueTypeInstance() is { } typeInstance)
		{
			if (typeInstance.TryGetValue(Type.ElementsLowercase, out var elementsValue) &&
				(elementsValue.IsList || elementsValue.IsText))
				return elementsValue.GetIteratorValue(executable.characterType, index);
			for (var valueIndex = 0; valueIndex < typeInstance.Values.Length; valueIndex++)
				if (typeInstance.Values[valueIndex].IsText)
					return typeInstance.Values[valueIndex].GetIteratorValue(executable.characterType, index);
		}
    return Memory.Frame.Get(GetFrameKey(listCall));
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
    var frameKey = GetFrameKey(memberCall);
		if (Memory.Frame.TryGet(frameKey, out var frameValue))
			return frameValue;
		if (Memory.Frame.TryGet(memberCall.Member.Name, out var memberFrameValue))
			return memberFrameValue;
		if (memberCall.Member.InitialValue is Value enumValue)
			return enumValue.Data;
   return new ValueInstance(frameKey);
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
		if (precompiledInstructions != null)
		{
			var evaluatedArguments = call.Arguments.Select(EvaluateExpression).ToArray();
			var evaluatedInstance = call.Instance != null
				? EvaluateExpression(call.Instance)
				: (ValueInstance?)null;
			var precompiledResult = RunChildScope(precompiledInstructions,
				() => InitializeMethodCallScope(call, evaluatedArguments, evaluatedInstance));
			return precompiledResult ?? new ValueInstance(call.Method.ReturnType, 0);
		}
		return TryEvaluateMethodCallFromBody(call) ??
			throw new InvalidOperationException(
				"No precompiled method instructions found for method call: " + call);
	}

	private ValueInstance? TryEvaluateMethodCallFromBody(MethodCall call)
	{
		var methodBody = call.Method.GetBodyAndParseIfNeeded();
		if (methodBody is not Body { Expressions.Count: > 0 } body)
			return null;
		var evaluatedInstance = call.Instance != null
			? EvaluateExpression(call.Instance)
			: (ValueInstance?)null;
		var savedFrame = Memory.Frame;
		Memory.Frame = new CallFrame(savedFrame);
		try
		{
			if (evaluatedInstance.HasValue)
			{
				Memory.Frame.Set(Type.ValueLowercase, evaluatedInstance.Value, isMember: true);
				var typeInstance = evaluatedInstance.Value.TryGetValueTypeInstance();
				if (typeInstance != null)
					TrySetScopeMembersFromTypeMembers(typeInstance);
			}
			for (var paramIndex = 0;
				paramIndex < call.Method.Parameters.Count && paramIndex < call.Arguments.Count;
				paramIndex++)
				Memory.Frame.Set(call.Method.Parameters[paramIndex].Name,
					EvaluateExpression(call.Arguments[paramIndex]));
			var lastExpression = body.Expressions[^1];
			return EvaluateExpression(lastExpression);
		}
		finally
		{
			Memory.Frame = savedFrame;
		}
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
      : Memory.Frame.Get(GetFrameKey(keyArg));
		var dictionary = Memory.Frame.Get(GetFrameKey(invoke.Method.Instance));
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
		instructionIndex = ExitExecutionLoopIndex;
		return true;
	}

	private const int ExitExecutionLoopIndex = 100_000;

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
			return;
		Memory.Frame.Set(Type.IndexLowercase, Memory.Frame.TryGet(Type.IndexLowercase, out var indexValue)
			? new ValueInstance(executable.numberType, indexValue.Number + 1)
			: new ValueInstance(executable.numberType, 0));
		if (!loopBegin.IsInitialized)
		{
			loopBegin.LoopCount = GetLength(iterableVariable);
			loopBegin.IsInitialized = true;
		}
		AlterValueVariable(iterableVariable, loopBegin);
		if (!string.IsNullOrEmpty(loopBegin.CustomVariableName))
			Memory.Frame.Set(loopBegin.CustomVariableName, Memory.Frame.Get(Type.ValueLowercase));
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
		var incrementValue = loopBegin.IsDecreasing == true
			? -1
			: 1;
		var currentIndex = Memory.Frame.TryGet(Type.IndexLowercase, out var indexValue)
			? indexValue.Number + incrementValue
			: loopBegin.StartIndexValue ?? 0;
		Memory.Frame.Set(Type.IndexLowercase, new ValueInstance(executable.numberType, currentIndex));
		Memory.Frame.Set(Type.ValueLowercase, Memory.Frame.Get(Type.IndexLowercase));
		if (!string.IsNullOrEmpty(loopBegin.CustomVariableName))
			Memory.Frame.Set(loopBegin.CustomVariableName,
				new ValueInstance(executable.numberType, currentIndex));
	}

	private static int GetLength(ValueInstance iterableInstance)
	{
		if (iterableInstance.IsText)
			return iterableInstance.Text.Length;
		if (iterableInstance.IsList)
			return iterableInstance.List.Items.Count;
		return (int)iterableInstance.Number;
	}

	private void AlterValueVariable(ValueInstance iterableVariable,
		LoopBeginInstruction loopBegin)
	{
		var index = (int)Memory.Frame.Get(Type.IndexLowercase).Number;
		if (iterableVariable.IsText)
		{
			if (index < iterableVariable.Text.Length)
				Memory.Frame.Set(Type.ValueLowercase,
					new ValueInstance(iterableVariable.Text[index].ToString()));
			return;
		}
		if (iterableVariable.IsList)
		{
			var items = iterableVariable.List.Items;
			if (index < items.Count)
				Memory.Frame.Set(Type.ValueLowercase, items[index]);
			else
				loopBegin.LoopCount = 0;
			return;
		}
		Memory.Frame.Set(Type.ValueLowercase,
			new ValueInstance(executable.numberType, index + 1));
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
			// Create a defensive copy to isolate the list state between separate Execute() calls
			// when lists are mutated in-place
			if (value.IsList)
				value = new ValueInstance(value.List.ReturnType, value.List.Items.ToArray());
			Memory.Frame.Set(storeVariable.Identifier, value, storeVariable.IsMember);
		}
		else if (instruction is StoreFromRegisterInstruction storeFromRegister)
		{
			if (!TryStoreToListElement(storeFromRegister))
				Memory.Frame.Set(storeFromRegister.Identifier,
					Memory.Registers[storeFromRegister.Register]);
		}
	}

	private bool TryStoreToListElement(StoreFromRegisterInstruction store)
	{
		var identifier = store.Identifier;
		var openParen = identifier.LastIndexOf('(');
		if (openParen <= 0 || !identifier.EndsWith(')'))
			return false;
		var listPath = identifier[..openParen];
		var indexExprName = identifier[(openParen + 1)..^1];
		if (!Memory.Frame.TryGet(listPath, out var listValue))
		{
			var lastDot = listPath.LastIndexOf('.');
			if (lastDot > 0)
				Memory.Frame.TryGet(listPath[(lastDot + 1)..], out listValue);
		}
		if (!listValue.IsList)
			return false;
		if (!Memory.Frame.TryGet(indexExprName, out var indexInstance))
			Memory.Frame.TryGet(Type.IndexLowercase, out indexInstance);
		if (!indexInstance.HasValue)
			return false;
		var index = (int)indexInstance.Number;
		if (index >= 0 && index < listValue.List.Items.Count)
		{
			listValue.List.Items[index] = Memory.Registers[store.Register];
			return true;
		}
		return false;
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
			_ => Memory.Registers[instruction.Registers[^1]]
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
			return new ValueInstance(ConvertToText(left).Text + ConvertToText(right).Text);
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
			_ => false
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
}