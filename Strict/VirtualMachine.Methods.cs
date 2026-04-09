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
		if (TryExecuteSpecialInvoke(invoke))
			return;
		var arguments = invoke.Method.Arguments;
		var evaluatedArgs = arguments.Count == 0
			? Array.Empty<ValueInstance>()
			: new ValueInstance[arguments.Count];
		for (var argIndex = 0; argIndex < arguments.Count; argIndex++)
			evaluatedArgs[argIndex] = EvaluateExpression(invoke.Method.Arguments[argIndex]);
		var evaluatedInstance = invoke.Method.Instance != null
			? EvaluateExpression(invoke.Method.Instance)
			: (ValueInstance?)null;
		var invokeInstructions = invoke.CachedInstructions ??=
			GetPrecompiledMethodInstructions(invoke) ??
			throw new InvalidOperationException("No precompiled method instructions found for " +
				invoke.Method.Method.Type.FullName + "." + invoke.Method.Method.Name +
				" with return type " + invoke.Method.ReturnType.FullName);
		var childScope = InitializeChildScope();
		InitializeMethodCallScope(invoke.Method, evaluatedArgs, evaluatedInstance);
		RunInstructions(invokeInstructions
#if DEBUG
			, invoke.Method.Method.Name
#endif
		);
		var result = TryFlattenNestedIteratorList(invoke.Method, Returns);
		CleanupChildScope(childScope);
		if (result != null)
			Memory.Registers[invoke.Register] = result.Value;
	}

	private static ValueInstance? TryFlattenNestedIteratorList(MethodCall methodCall,
		ValueInstance? result)
	{
		if (result == null || methodCall.Method.Name != Keyword.For ||
			!methodCall.Method.ReturnType.IsIterator || !result.Value.IsList)
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
		var methodCall = invoke.Method;
		var instanceExpression = methodCall.Instance;
		return methodCall.Method.Name switch
		{
			Method.From => ExecuteFromInvoke(invoke, methodCall.ReturnType),
			BinaryOperator.To => instanceExpression != null && TryHandleToConversion(invoke),
			"Length" or "Count" => instanceExpression != null && TryHandleNativeLength(invoke),
			"Increment" or "Decrement" => TryHandleIncrementDecrement(invoke),
			"Get" => instanceExpression != null && instanceExpression.ReturnType.IsDictionary &&
				GetValueByKeyForDictionaryAndStoreInRegister(invoke),
			"StartsWith" or "IndexOf" or "Substring" => TryHandleNativeTextMethod(invoke),
			_ => instanceExpression is MemberCall memberCall &&
				memberCall.Member.Type.Name is Type.Logger or Type.TextWriter or Type.System &&
				TryHandleNativeTraitMethod(invoke)
		};
	}

	private bool ExecuteFromInvoke(Invoke invoke, Type returnType)
	{
		if (returnType.IsDictionary)
		{
			Memory.Registers[invoke.Register] = new ValueInstance(returnType,
				new Dictionary<ValueInstance, ValueInstance>());
			return true;
		}
		return TryHandleFromConstructor(invoke, returnType);
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

	private bool TryHandleNativeLength(Invoke invoke)
	{
		var instanceExpression = invoke.Method.Instance;
		if (instanceExpression == null)
			return false;
		var instanceValue = instanceExpression is MemberCall memberCall
			? EvaluateMemberCall(memberCall)
			: EvaluateExpression(instanceExpression);
		if (!TryGetNativeLength(instanceValue, invoke.Method.Method.Name, out var lengthValue))
			return false;
		Memory.Registers[invoke.Register] = lengthValue;
		return true;
	}

	private bool TryHandleIncrementDecrement(Invoke invoke)
	{
		var methodName = invoke.Method.Method.Name;
		if (methodName != "Increment" && methodName != "Decrement")
			return false;
		if (invoke.Method.Instance == null)
			return false;
		var current = EvaluateExpression(invoke.Method.Instance);
		var delta = methodName == "Increment"
			? 1.0
			: -1.0;
		Memory.Registers[invoke.Register] =
			new ValueInstance(current.GetType(), current.Number + delta);
		return true;
	}

	//TODO: find all [.. with existing list and no changes, all those cases need to be removed, there is a crazy amount of those added (54 wtf)!
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

	private List<Instruction>? GetPrecompiledMethodInstructions(Invoke invoke) =>
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
			//TODO: this seems to be more of a hack
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

	private ChildScopeState InitializeChildScope()
	{
		var savedInstructions = instructions;
		var savedIndex = instructionIndex;
		var savedConditionFlag = conditionFlag;
		var savedReturns = Returns;
		var savedFrame = Memory.Frame;
		var depth = registerStackDepth++;
		//TODO: needs testing and cleanU
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
		return new ValueInstance(executable.booleanType, matches);
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
}
