using System.Diagnostics;
using System.Globalization;
using Strict.Bytecode;
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
		var implicitInstance = TryGetImplicitInstance(invoke);
		if (TryExecuteSpecialInvoke(invoke, implicitInstance))
			return;
		var info = invoke.MethodInfo;
		var evaluatedArgs = info.ArgumentRegisters.Length == 0
			? Array.Empty<ValueInstance>()
			: new ValueInstance[info.ArgumentRegisters.Length];
		for (var argIndex = 0; argIndex < info.ArgumentRegisters.Length; argIndex++)
			evaluatedArgs[argIndex] = Memory.Registers[info.ArgumentRegisters[argIndex]];
		var evaluatedInstance = info.InstanceRegister.HasValue
			? Memory.Registers[info.InstanceRegister.Value]
			: implicitInstance;
		var invokeInstructions = invoke.CachedInstructions ??=
			GetPrecompiledMethodInstructions(invoke) ?? throw Fail(
				"No precompiled method instructions found for '" + info.TypeFullName + "." +
				info.MethodName + "' with return type " + info.ReturnTypeName);
		var childScope = InitializeChildScope();
		var previousMethodContext = currentMethodContext;
		var previousInstance = currentInstance;
		currentMethodContext = info.FullName;
		currentInstance = evaluatedInstance;
		InitializeMethodCallScope(info, evaluatedArgs, evaluatedInstance);
		var isReentrant = !runningBlocks.Add(invokeInstructions);
		var savedLoops = isReentrant
			? SaveLoopStates(invokeInstructions)
			: null;
		var started = Profile == null
			? 0
			: Stopwatch.GetTimestamp();
		RunInstructions(invokeInstructions
#if DEBUG
			, info.MethodName
#endif
		);
		if (Profile != null)
			AddToProfile(info.FullName, Stopwatch.GetElapsedTime(started));
		if (savedLoops == null)
			runningBlocks.Remove(invokeInstructions);
		else
			foreach (var (loopBegin, state) in savedLoops)
				loopBegin.RestoreState(state);
		var result = TryFlattenNestedIteratorList(info, Returns);
		currentMethodContext = previousMethodContext;
		currentInstance = previousInstance;
		CleanupChildScope(childScope);
		if (result != null)
			Memory.Registers[invoke.Register] = result.Value;
	}

	/// <summary>
	/// Optional inclusive time and call count per invoked method, used by the -profile option.
	/// </summary>
	public Dictionary<string, MethodTime>? Profile { get; init; }

	public readonly record struct MethodTime(TimeSpan Time, int Calls);

	private void AddToProfile(string methodName, TimeSpan elapsed)
	{
		var previous = Profile!.GetValueOrDefault(methodName);
		Profile[methodName] = new MethodTime(previous.Time + elapsed, previous.Calls + 1);
	}

	/// <summary>
	/// Mirrors HighLevelRuntime semantics: when a method call has no explicit instance
	/// (and is not a `from` constructor), use the surrounding frame's `value` as the
	/// implicit instance. Without this, sibling instance calls inside base-type methods
	/// like <c>Text.Replace</c> calling <c>IndexOf</c> would lose their `value` and
	/// recurse forever because the bail-out `if separatorIndex is -1 return value` never
	/// fires (IndexOf returns 0 instead of -1 for an empty/missing instance).
	/// </summary>
	private ValueInstance? TryGetImplicitInstance(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		if (info.InstanceRegister.HasValue || info.MethodName == Method.From)
			return null;
		if (currentInstance.HasValue)
			return currentInstance;
		return Memory.Frame.TryGet(ValueSymbolId, out var implicitInstance)
			? implicitInstance
			: null;
	}

	private static ValueInstance? TryFlattenNestedIteratorList(InvokeMethodInfo info,
		ValueInstance? result)
	{
		if (result == null || info.MethodName != Keyword.For || !result.Value.IsList)
			return result;
		var materialized = result.Value;
		if (materialized.List.Items.Count == 0 || !materialized.List.Items.All(item => item.IsList))
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

	private bool TryExecuteSpecialInvoke(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var hasInstance = info.InstanceRegister.HasValue || implicitInstance != null;
		if (!hasInstance && TryHandleNativeTraitStaticMethod(invoke))
			return true;
		if (!info.InstanceRegister.HasValue && TryHandleNativeStaticTypeMethod(invoke))
			return true;
		if (hasInstance && TryHandleNativeProcessInstanceMethod(invoke, implicitInstance))
			return true;
		return info.MethodName switch
		{
			Method.From => ExecuteFromInvoke(invoke, info.ResolveReturnType(executable.TypeResolver)),
			BinaryOperator.To => hasInstance && TryHandleToConversion(invoke, implicitInstance),
			"Length" or "Count" => hasInstance && info.ArgumentRegisters.Length == 0 &&
				TryHandleNativeLength(invoke, implicitInstance),
			BinaryOperator.In or "Index" => hasInstance && TryHandleNativeListSearch(invoke, implicitInstance),
			"ReadLines" or "ReadBytes" or "Write" or "Delete" or "Exists" or "Close" => hasInstance &&
				TryHandleNativeFileMethod(invoke, implicitInstance),
			"Floor" => hasInstance && TryHandleNativeFloor(invoke, implicitInstance),
			"Increment" => TryHandleIncrementDecrement(invoke, true, implicitInstance),
			"Decrement" => TryHandleIncrementDecrement(invoke, false, implicitInstance),
			"StartsWith" or "IndexOf" or "LastIndexOf" or "Substring" or "Upper" or "Lower"
				or "Capitalize" or "Trim" or "TrimStart"
				or "TrimEnd" => hasInstance && TryHandleNativeTextMethod(invoke, implicitInstance),
			// Avoid infinite recursion when Boolean.strict operators are compiled as Invoke
			// (their .strict bodies historically used the same operators recursively).
			BinaryOperator.And or BinaryOperator.Or or BinaryOperator.Xor or "not" => hasInstance &&
				TryHandleNativeBooleanMethod(invoke, implicitInstance),
			_ => (info.InstanceRegister.HasValue || implicitInstance != null) &&
				TryHandleNativeTraitInstanceMethod(invoke, implicitInstance)
		};
	}

	private ValueInstance ResolveInvokeInstance(InvokeMethodInfo info,
		ValueInstance? implicitInstance) =>
		info.InstanceRegister.HasValue
			? Memory.Registers[info.InstanceRegister.Value]
			: implicitInstance!.Value;

	private bool ExecuteFromInvoke(Invoke invoke, Type returnType)
	{
		if (returnType.Name == Type.File)
		{
			if (invoke.MethodInfo.ArgumentRegisters.Length != 1)
				return false;
			if (!FileValue.TryGetPathText(Memory.Registers[invoke.MethodInfo.ArgumentRegisters[0]],
				out var pathText))
				return false;
			var fileInstance = NativeFileRegistry.Open(returnType, pathText);
			Memory.Frame.TrackDisposable(fileInstance);
			Memory.Registers[invoke.Register] = fileInstance;
			return true;
		}
		if (returnType.IsDictionary)
		{
			Memory.Registers[invoke.Register] = new ValueInstance(returnType,
				new Dictionary<ValueInstance, ValueInstance>());
			return true;
		}
		if (returnType.IsList && invoke.MethodInfo.ArgumentRegisters.Length == 0)
		{
			Memory.Registers[invoke.Register] =
				new ValueInstance(returnType, Array.Empty<ValueInstance>());
			return true;
		}
		if (!returnType.IsMutable && (returnType.IsNumber || returnType.IsText ||
			returnType.IsCharacter || returnType.IsEnum || returnType.IsBoolean || returnType.IsNone))
		{
			var info = invoke.MethodInfo;
			Memory.Registers[invoke.Register] = info.ArgumentRegisters.Length > 0
				? ToConstructedValue(returnType, Memory.Registers[info.ArgumentRegisters[0]])
				: CreateDefaultValue(returnType);
			return true;
		}
		if (IsTrait(returnType) && TryCallNativeFromPlugin(invoke, returnType))
			return true;
		return TryHandleFromConstructor(invoke, returnType);
	}

	private ValueInstance ToConstructedValue(Type returnType, ValueInstance argument) =>
		returnType.IsCharacter
			? new ValueInstance(executable.characterType, argument.IsText
				? argument.Text[0]
				: argument.Number)
			: argument;

	private List<Instruction>? GetPrecompiledMethodInstructions(Method method) =>
		executable.FindInstructions(method.Type, method) ??
		executable.FindInstructions(method.Type.Name, method.Name, method.Parameters.Count,
			method.ReturnType.Name) ??
		executable.FindInstructions(nameof(Strict) + Context.ParentSeparator + method.Type.Name,
			method.Name, method.Parameters.Count, method.ReturnType.Name) ??
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
			executable.FindInstructions(nameof(Strict) + Context.ParentSeparator + info.TypeFullName,
				info.MethodName, info.ParameterNames.Length, info.ReturnTypeName) ??
			FindInstructionsFromInvokeInfo(info);
	}

	private List<Instruction>? FindInstructionsFromInvokeInfo(InvokeMethodInfo info)
	{
		var strictPrefix = nameof(Strict) + Context.ParentSeparator;
		if (info.TypeFullName.StartsWith(strictPrefix, StringComparison.Ordinal))
			return executable.FindInstructions(info.TypeFullName[strictPrefix.Length..], info.MethodName,
				info.ParameterNames.Length, info.ReturnTypeName) ?? FindInstructionsByTypeSuffix(info);
		return executable.FindInstructions(strictPrefix + info.TypeFullName, info.MethodName,
			info.ParameterNames.Length, info.ReturnTypeName) ?? FindInstructionsByTypeSuffix(info);
	}

	private List<Instruction>? FindInstructionsByTypeSuffix(InvokeMethodInfo info)
	{
		var typeFullName = info.TypeFullName;
		var strictPrefix = nameof(Strict) + Context.ParentSeparator;
		for (var separatorIndex = typeFullName.IndexOf(Context.ParentSeparator); separatorIndex >= 0;
			separatorIndex = typeFullName.IndexOf(Context.ParentSeparator, separatorIndex + 1))
		{
			var strippedTypeName = typeFullName[(separatorIndex + 1)..];
			var foundInstructions =
				executable.FindInstructions(strippedTypeName, info.MethodName, info.ParameterNames.Length,
					info.ReturnTypeName) ?? executable.FindInstructions(strictPrefix + strippedTypeName,
					info.MethodName, info.ParameterNames.Length, info.ReturnTypeName);
			if (foundInstructions != null)
				return foundInstructions;
		}
		return null;
	}

	/// <summary>
	/// Parameters are bound after the instance members: a method called on a type name keeps the
	/// caller's implicit instance, its parameters must win over caller members of the same name.
	/// </summary>
	private void InitializeMethodCallScope(InvokeMethodInfo info, ValueInstance[] evaluatedArguments,
		ValueInstance? evaluatedInstance)
	{
		if (evaluatedInstance.HasValue)
			SetInstanceScope(evaluatedInstance.Value);
		for (var parameterIndex = 0; parameterIndex < info.ParameterNames.Length &&
			parameterIndex < evaluatedArguments.Length; parameterIndex++)
			Memory.Frame.Set(info.ParameterNames[parameterIndex], evaluatedArguments[parameterIndex]);
		for (var parameterIndex = evaluatedArguments.Length;
			parameterIndex < info.ParameterNames.Length; parameterIndex++)
			Memory.Frame.Set(info.ParameterNames[parameterIndex],
				new ValueInstance(executable.numberType, 0.0));
	}

	private void SetInstanceScope(ValueInstance instance)
	{
		Memory.Frame.Set(Type.ValueLowercase, instance, true);
		if (instance.IsText || instance.IsList)
		{
			Memory.Frame.Set(CallFrame.ElementsSymbolId, instance, true);
			if (instance.IsText)
				Memory.Frame.Set("characters", instance, true);
			return;
		}
		var flatNumeric = instance.TryGetFlatNumericArrayInstance();
		if (flatNumeric != null)
		{
			var flatMembers = flatNumeric.ReturnType.Members;
			for (var memberIndex = 0; memberIndex < flatMembers.Count &&
				memberIndex < flatNumeric.FlatWidth; memberIndex++)
				if (!IsTrait(flatMembers[memberIndex].Type))
					Memory.Frame.Set(flatMembers[memberIndex].Name,
						new ValueInstance(flatMembers[memberIndex].Type, flatNumeric.GetFlat(memberIndex)),
						true);
			return;
		}
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null && TrySetScopeMembersFromTypeMembers(typeInstance))
			return;
		// Boolean/Number/None primitives: only `value` is needed (already set above).
		if (instance.IsPrimitiveType(executable.booleanType) ||
			instance.IsPrimitiveType(executable.numberType) ||
			instance.IsPrimitiveType(executable.noneType))
			return;
		Type instanceType;
		try
		{
			instanceType = instance.GetType();
		}
		catch (InvalidCastException)
		{
			return;
		}
		if (instanceType == null)
			return;
		var firstNonTraitMember = instanceType.Members.FirstOrDefault(member => !IsTrait(member.Type));
		if (firstNonTraitMember != null)
			Memory.Frame.Set(firstNonTraitMember.Name, instance, true);
	}

	private bool TrySetScopeMembersFromTypeMembers(ValueTypeInstance typeInstance)
	{
		var members = typeInstance.ReturnType.Members;
		if (members.Count > 0)
		{
			for (var memberIndex = 0; memberIndex < members.Count &&
				memberIndex < typeInstance.Values.Length; memberIndex++)
				if (!IsTrait(members[memberIndex].Type) || typeInstance.Values[memberIndex].HasValue)
					Memory.Frame.Set(members[memberIndex].Name, typeInstance.Values[memberIndex], true);
			return true;
		}
		if (!TryGetBinaryMembers(typeInstance.ReturnType, out var binaryMembers) ||
			binaryMembers.Count == 0)
			return false;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count &&
			memberIndex < typeInstance.Values.Length; memberIndex++)
			Memory.Frame.Set(binaryMembers[memberIndex].Name, typeInstance.Values[memberIndex], true);
		return true;
	}

	/// <summary>
	/// Type.IsTrait walks members and methods each time, types no longer change while executing.
	/// </summary>
	private bool IsTrait(Type type)
	{
		if (!isTraitPerType.TryGetValue(type, out var isTrait))
			isTraitPerType[type] = isTrait = type.IsTrait;
		return isTrait;
	}

	private readonly Dictionary<Type, bool> isTraitPerType = new();

	private readonly HashSet<List<Instruction>> runningBlocks = new(ReferenceEqualityComparer.Instance);

	private static List<(LoopBeginInstruction, LoopBeginInstruction.State)>? SaveLoopStates(
		List<Instruction> blockInstructions)
	{
		List<(LoopBeginInstruction, LoopBeginInstruction.State)>? states = null;
		foreach (var instruction in blockInstructions)
			if (instruction is LoopBeginInstruction loopBegin)
				(states ??= []).Add((loopBegin, loopBegin.SaveState()));
		return states ?? [];
	}

	private bool TryGetBinaryMembers(Type type, out List<BinaryMember> members)
	{
		if (!binaryMembersPerType.TryGetValue(type, out var cached))
			binaryMembersPerType[type] = cached = FindBinaryMembers(type);
		members = cached ?? [];
		return cached != null;
	}

	private readonly Dictionary<Type, List<BinaryMember>?> binaryMembersPerType = new();

	private List<BinaryMember>? FindBinaryMembers(Type type)
	{
		foreach (var (typeName, typeData) in executable.MethodsPerType)
			if (typeData.Members.Count > 0 && (typeName == type.FullName || typeName == type.Name ||
				typeName.EndsWith(Context.ParentSeparator + type.Name, StringComparison.Ordinal)))
				return typeData.Members;
		return null;
	}

	private ChildScopeState InitializeChildScope()
	{
		var savedInstructions = instructions;
		var savedIndex = instructionIndex;
		var savedConditionFlag = conditionFlag;
		var savedReturns = Returns;
		var savedFrame = Memory.Frame;
		if (registerStackDepth >= MaxCallDepth ||
			!System.Runtime.CompilerServices.RuntimeHelpers.TryEnsureSufficientExecutionStack())
			throw new StackOverflow(registerStackDepth, currentMethodContext);
		var depth = registerStackDepth++;
		// ReSharper disable once ConvertIfStatementToNullCoalescingAssignment
		// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
		if (registerStack[depth] == null)
			registerStack[depth] = new ValueInstance[Registers.Count];
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
		DisposeTrackedValues(state.Frame, Returns, state.SavedFrame);
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

	private readonly record struct ChildScopeState(List<Instruction> SavedInstructions,
		int SavedInstructionIndex,
		bool SavedConditionFlag,
		ValueInstance? SavedReturns,
		CallFrame SavedFrame,
		int StackDepth,
		CallFrame Frame);

	private void DisposeTrackedValues(CallFrame frame, ValueInstance? returnValue,
		CallFrame? parentFrame)
	{
		foreach (var value in frame.DisposableValues.ToArray())
			if (returnValue.HasValue && value.Equals(returnValue.Value))
			{
				if (parentFrame != null)
				{
					parentFrame.TrackDisposable(value);
					frame.RemoveDisposable(value);
				}
			}
			else if (FileValue.TryGetHandle(value, executable.TypeResolver.GetType(Type.File),
				out var handle))
			{
				NativeFileRegistry.Close(handle);
			}
	}
}
