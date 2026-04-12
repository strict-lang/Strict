using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	private void ExecuteInvoke(Invoke invoke)
	{
		if (TryExecuteSpecialInvoke(invoke))
			return;
		var info = invoke.MethodInfo;
		var evaluatedArgs = info.ArgumentRegisters.Length == 0
			? Array.Empty<ValueInstance>()
			: new ValueInstance[info.ArgumentRegisters.Length];
		for (var argIndex = 0; argIndex < info.ArgumentRegisters.Length; argIndex++)
			evaluatedArgs[argIndex] = Memory.Registers[info.ArgumentRegisters[argIndex]];
		var evaluatedInstance = info.InstanceRegister.HasValue
			? Memory.Registers[info.InstanceRegister.Value]
			: (ValueInstance?)null;
		var invokeInstructions = invoke.CachedInstructions ??=
			GetPrecompiledMethodInstructions(invoke) ??
			throw Fail("No precompiled method instructions found for '" +
				info.TypeFullName + "." + info.MethodName +
				"' with return type " + info.ReturnTypeName);
		var childScope = InitializeChildScope();
		var previousMethodContext = currentMethodContext;
		currentMethodContext = info.TypeFullName + "." + info.MethodName;
		if (info.MethodName == "Process")
		{
			Console.Error.WriteLine("DEBUG Process invoke: params=" + string.Join(",", info.ParameterNames) +
				" argRegs=" + string.Join(",", info.ArgumentRegisters) +
				" args=" + string.Join(",", evaluatedArgs.Select(arg => arg.HasValue + ":" + arg)) +
				" instance=" + evaluatedInstance?.HasValue + ":" + evaluatedInstance);
		}
		InitializeMethodCallScope(info, evaluatedArgs, evaluatedInstance);
		if (info.MethodName is "Process" or "for")
		{
			Console.Error.WriteLine("DEBUG " + info.TypeFullName + "." + info.MethodName +
				" params: " + string.Join(",", info.ParameterNames));
		}
		RunInstructions(invokeInstructions
#if DEBUG
			, info.MethodName
#endif
		);
		var result = TryFlattenNestedIteratorList(info, Returns);
		currentMethodContext = previousMethodContext;
		CleanupChildScope(childScope);
		if (result != null)
			Memory.Registers[invoke.Register] = result.Value;
	}

	private static ValueInstance? TryFlattenNestedIteratorList(InvokeMethodInfo info,
		ValueInstance? result)
	{
		if (result == null || info.MethodName != Keyword.For || !result.Value.IsList)
			return result;
		var materialized = result.Value;
		if (materialized.List.Items.Count == 0 ||
			!materialized.List.Items.All(item => item.IsList))
			return result;
		var flattenedItems = new List<ValueInstance>();
		foreach (var nested in materialized.List.Items)
			flattenedItems.AddRange(nested.List.Items);
		if (flattenedItems.Count == 0)
			return result;
		var flattenedElementType = flattenedItems[0].GetType();
		return new ValueInstance(materialized.GetType().GetGenericImplementation(flattenedElementType),
			flattenedItems.ToArray());
	}

	private bool TryExecuteSpecialInvoke(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		if (info.MethodName == Method.From && !info.ReturnTypeName.Contains("Number"))
			Console.Error.WriteLine("DEBUG TrySpecial from: type=" + info.TypeFullName +
				" ret=" + info.ReturnTypeName + " params=" + string.Join(",", info.ParameterNames));
		return info.MethodName switch
		{
			Method.From => ExecuteFromInvoke(invoke, info.ResolveReturnType(executable.basePackage)),
			BinaryOperator.To => info.InstanceRegister.HasValue && TryHandleToConversion(invoke),
			"Length" or "Count" => info.InstanceRegister.HasValue && TryHandleNativeLength(invoke),
			"Increment" => TryHandleIncrementDecrement(invoke, isIncrement: true),
			"Decrement" => TryHandleIncrementDecrement(invoke, isIncrement: false),
			"StartsWith" or "IndexOf" or "Substring" => TryHandleNativeTextMethod(invoke),
			_ => false
		};
	}

	private bool ExecuteFromInvoke(Invoke invoke, Type returnType)
	{
		if (returnType.Name == "ColorImage")
			Console.Error.WriteLine("DEBUG ExecuteFromInvoke ColorImage: isMutable=" + returnType.IsMutable +
				" isList=" + returnType.IsList + " isDict=" + returnType.IsDictionary +
				" members=" + returnType.Members.Count);
		if (returnType.IsDictionary)
		{
			Memory.Registers[invoke.Register] = new ValueInstance(returnType,
				new Dictionary<ValueInstance, ValueInstance>());
			return true;
		}
		if (!returnType.IsMutable && (returnType.IsNumber || returnType.IsText ||
			returnType.IsCharacter || returnType.IsEnum || returnType.IsBoolean || returnType.IsNone))
		{
			var info = invoke.MethodInfo;
			Memory.Registers[invoke.Register] = info.ArgumentRegisters.Length > 0
				? Memory.Registers[info.ArgumentRegisters[0]]
				: CreateDefaultValue(returnType);
			return true;
		}
		return TryHandleFromConstructor(invoke, returnType);
	}

	private bool TryHandleToConversion(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		var conversionType = info.ResolveReturnType(executable.basePackage);
		var rawValue = Memory.Registers[info.InstanceRegister!.Value];
		if (conversionType.IsText)
			Memory.Registers[invoke.Register] = ConvertToText(rawValue);
		else if (conversionType.IsNumber)
			Memory.Registers[invoke.Register] =
				rawValue.IsText
					? new ValueInstance(conversionType, Convert.ToDouble(rawValue.Text))
					: rawValue;
		return true;
	}

	private bool TryHandleNativeLength(Invoke invoke)
	{
		var instanceValue = Memory.Registers[invoke.MethodInfo.InstanceRegister!.Value];
		if (!TryGetNativeLength(instanceValue, invoke.MethodInfo.MethodName, out var lengthValue))
			return false;
		Memory.Registers[invoke.Register] = lengthValue;
		return true;
	}

	private bool TryHandleIncrementDecrement(Invoke invoke, bool isIncrement)
	{
		var current = Memory.Registers[invoke.MethodInfo.InstanceRegister!.Value];
		var delta = isIncrement
			? 1.0
			: -1.0;
		Memory.Registers[invoke.Register] =
			new ValueInstance(current.GetType(), current.Number + delta);
		return true;
	}

	private List<Instruction>? GetPrecompiledMethodInstructions(Method method) =>
		executable.FindInstructions(method.Type, method) ??
		executable.FindInstructions(method.Type.Name, method.Name,
			method.Parameters.Count, method.ReturnType.Name) ??
		executable.FindInstructions(
			nameof(Strict) + Context.ParentSeparator + method.Type.Name, method.Name,
			method.Parameters.Count, method.ReturnType.Name) ??
		FindInstructionsWithMissingRootPackagePrefix(method) ??
		FindInstructionsWithStrippedPackagePrefix(method);

	private List<Instruction>? FindInstructionsWithMissingRootPackagePrefix(Method method)
	{
		var fullName = method.Type.FullName;
		return fullName.StartsWith(nameof(Strict) + Context.ParentSeparator, StringComparison.Ordinal)
			? null
			: executable.FindInstructions(nameof(Strict) + Context.ParentSeparator + fullName,
				method.Name, method.Parameters.Count, method.ReturnType.Name);
	}

	private List<Instruction>? FindInstructionsWithStrippedPackagePrefix(Method method)
	{
		var fullName = method.Type.FullName;
		var strictPrefix = nameof(Strict) + Context.ParentSeparator;
		return fullName.StartsWith(strictPrefix, StringComparison.Ordinal)
			? executable.FindInstructions(fullName[strictPrefix.Length..], method.Name,
				method.Parameters.Count, method.ReturnType.Name)
			: null;
	}

	private List<Instruction>? GetPrecompiledMethodInstructions(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		return executable.FindInstructions(info.TypeFullName, info.MethodName,
				info.ParameterNames.Length, info.ReturnTypeName) ??
			executable.FindInstructions(
				nameof(Strict) + Context.ParentSeparator + info.TypeFullName, info.MethodName,
				info.ParameterNames.Length, info.ReturnTypeName) ??
			FindInstructionsFromInvokeInfo(info);
	}

	private List<Instruction>? FindInstructionsFromInvokeInfo(InvokeMethodInfo info)
	{
		var strictPrefix = nameof(Strict) + Context.ParentSeparator;
		if (info.TypeFullName.StartsWith(strictPrefix, StringComparison.Ordinal))
			return executable.FindInstructions(info.TypeFullName[strictPrefix.Length..],
				info.MethodName, info.ParameterNames.Length, info.ReturnTypeName) ??
				FindInstructionsByTypeSuffix(info);
		return executable.FindInstructions(strictPrefix + info.TypeFullName,
				info.MethodName, info.ParameterNames.Length, info.ReturnTypeName) ??
			FindInstructionsByTypeSuffix(info);
	}

	private List<Instruction>? FindInstructionsByTypeSuffix(InvokeMethodInfo info)
	{
		var typeFullName = info.TypeFullName;
		var strictPrefix = nameof(Strict) + Context.ParentSeparator;
		for (var separatorIndex = typeFullName.IndexOf(Context.ParentSeparator);
			separatorIndex >= 0; separatorIndex = typeFullName.IndexOf(Context.ParentSeparator,
				separatorIndex + 1))
		{
			var strippedTypeName = typeFullName[(separatorIndex + 1)..];
			var instructions = executable.FindInstructions(strippedTypeName, info.MethodName,
				info.ParameterNames.Length, info.ReturnTypeName) ??
				executable.FindInstructions(strictPrefix + strippedTypeName, info.MethodName,
					info.ParameterNames.Length, info.ReturnTypeName);
			if (instructions != null)
				return instructions;
		}
		return null;
	}

	private void InitializeMethodCallScope(InvokeMethodInfo info,
		IReadOnlyList<ValueInstance> evaluatedArguments, ValueInstance? evaluatedInstance)
	{
		for (var parameterIndex = 0; parameterIndex < info.ParameterNames.Length &&
			parameterIndex < evaluatedArguments.Count; parameterIndex++)
			Memory.Frame.Set(info.ParameterNames[parameterIndex],
				evaluatedArguments[parameterIndex]);
		if (!evaluatedInstance.HasValue)
			return;
		var instance = evaluatedInstance.Value;
		Memory.Frame.Set(Type.ValueLowercase, instance, isMember: true);
		if (instance.IsText)
		{
			Memory.Frame.Set("elements", instance, isMember: true);
			Memory.Frame.Set("characters", instance, isMember: true);
			return;
		}
		var flatNumeric = instance.TryGetFlatNumericArrayInstance();
		if (flatNumeric != null)
		{
			var flatMembers = flatNumeric.ReturnType.Members;
			for (var memberIndex = 0; memberIndex < flatMembers.Count &&
				memberIndex < flatNumeric.FlatWidth; memberIndex++)
				if (!flatMembers[memberIndex].Type.IsTrait)
					Memory.Frame.Set(flatMembers[memberIndex].Name,
						new ValueInstance(flatMembers[memberIndex].Type, flatNumeric.GetFlat(memberIndex)),
						isMember: true);
			return;
		}
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null && TrySetScopeMembersFromTypeMembers(typeInstance))
			return;
		var firstNonTraitMember = instance.GetType().Members.FirstOrDefault(member =>
			!member.Type.IsTrait);
		if (firstNonTraitMember != null)
			Memory.Frame.Set(firstNonTraitMember.Name, instance, isMember: true);
	}

	private bool TrySetScopeMembersFromTypeMembers(ValueTypeInstance typeInstance)
	{
		var members = typeInstance.ReturnType.Members;
		if (members.Count > 0)
		{
			for (var memberIndex = 0; memberIndex < members.Count &&
				memberIndex < typeInstance.Values.Length; memberIndex++)
				if (!members[memberIndex].Type.IsTrait)
					Memory.Frame.Set(members[memberIndex].Name, typeInstance.Values[memberIndex],
						isMember: true);
			return true;
		}
		if (!TryGetBinaryMembers(typeInstance.ReturnType, out var binaryMembers) ||
			binaryMembers.Count == 0)
			return false;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count &&
			memberIndex < typeInstance.Values.Length; memberIndex++)
		{
			var memberType = typeInstance.ReturnType.FindType(binaryMembers[memberIndex].FullTypeName) ??
				typeInstance.ReturnType.FindType(GetShortTypeName(binaryMembers[memberIndex].FullTypeName));
			if (memberType is { IsTrait: true })
				continue;
			Memory.Frame.Set(binaryMembers[memberIndex].Name, typeInstance.Values[memberIndex],
				isMember: true);
		}
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

	private ChildScopeState InitializeChildScope()
	{
		var savedInstructions = instructions;
		var savedIndex = instructionIndex;
		var savedConditionFlag = conditionFlag;
		var savedReturns = Returns;
		var savedFrame = Memory.Frame;
		var depth = registerStackDepth++;
		// ReSharper disable once ConvertIfStatementToNullCoalescingAssignment
		// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
		if (registerStack[depth] == null)
			registerStack[depth] = new ValueInstance[16];
		Memory.Registers.SaveTo(registerStack[depth]);
		var frame = framePoolDepth > 0
			? framePool[--framePoolDepth]
			: new CallFrame();
		frame.Reset(savedFrame);
		Memory.Frame = frame;
		Returns = null;
		return new ChildScopeState(savedInstructions, savedIndex, savedConditionFlag, savedReturns,
			savedFrame, depth, frame);
	}

	private void CleanupChildScope(ChildScopeState state)
	{
		state.Frame.Reset(null);
		if (framePoolDepth < MaxCallDepth)
			framePool[framePoolDepth++] = state.Frame;
		Memory.Frame = state.SavedFrame;
		registerStackDepth = state.StackDepth;
		Memory.Registers.RestoreFrom(registerStack[state.StackDepth]);
		instructions = state.SavedInstructions;
		instructionIndex = state.SavedInstructionIndex;
		conditionFlag = state.SavedConditionFlag;
		Returns = state.SavedReturns;
	}

	private readonly record struct ChildScopeState(IReadOnlyList<Instruction> SavedInstructions,
		int SavedInstructionIndex, bool SavedConditionFlag, ValueInstance? SavedReturns,
		CallFrame SavedFrame, int StackDepth, CallFrame Frame);

	private bool TryHandleNativeTextMethod(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		var instance = Memory.Registers[info.InstanceRegister!.Value];
		if (!instance.IsText)
			return false;
		var text = instance.Text;
		var args = new ValueInstance[info.ArgumentRegisters.Length];
		for (var argIndex = 0; argIndex < info.ArgumentRegisters.Length; argIndex++)
			args[argIndex] = Memory.Registers[info.ArgumentRegisters[argIndex]];
		Memory.Registers[invoke.Register] = info.MethodName switch
		{
			"StartsWith" => EvaluateStartsWith(text, args),
			"IndexOf" => new ValueInstance(executable.numberType,
				text.IndexOf(args[0].Text, StringComparison.Ordinal)),
			"Substring" => new ValueInstance(
				text.Substring((int)args[0].Number, (int)args[1].Number)),
			_ => throw new InvalidOperationException("Unhandled native text method: " + info.MethodName)
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
		return new ValueInstance(executable.booleanType, matches);
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
				result = new ValueInstance(executable.numberType, instance.List.Count);
				return true;
			}
		}
		result = default;
		return false;
	}

	internal static ValueInstance ConvertToText(ValueInstance rawValue)
	{
		if (rawValue.IsText)
			return rawValue;
		if (rawValue.TryGetValueTypeInstance() is { } typeInstance)
			return new ValueInstance(typeInstance.ToAutomaticText());
		return new ValueInstance(rawValue.ToExpressionCodeString());
	}
}