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
			GetPrecompiledMethodInstructions(invoke) ??
			throw Fail("No precompiled method instructions found for '" +
				info.TypeFullName + "." + info.MethodName +
				"' with return type " + info.ReturnTypeName);
		var childScope = InitializeChildScope();
		var previousMethodContext = currentMethodContext;
		currentMethodContext = info.TypeFullName + "." + info.MethodName;
		InitializeMethodCallScope(info, evaluatedArgs, evaluatedInstance);
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

	private bool TryExecuteSpecialInvoke(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var hasInstance = info.InstanceRegister.HasValue || implicitInstance != null;
		return info.MethodName switch
		{
			Method.From => ExecuteFromInvoke(invoke, info.ResolveReturnType(executable.basePackage)),
			BinaryOperator.To => hasInstance && TryHandleToConversion(invoke, implicitInstance),
			"Length" or "Count" => hasInstance && TryHandleNativeLength(invoke, implicitInstance),
			"ReadText" or "ReadBytes" or "Write" or "Delete" or "Exists" or "Close" =>
				hasInstance && TryHandleNativeFileMethod(invoke, implicitInstance),
			"Increment" => TryHandleIncrementDecrement(invoke, isIncrement: true, implicitInstance),
			"Decrement" => TryHandleIncrementDecrement(invoke, isIncrement: false, implicitInstance),
			"StartsWith" or "IndexOf" or "LastIndexOf" or "Substring" or "Upper" or "Lower" =>
				hasInstance && TryHandleNativeTextMethod(invoke, implicitInstance),
			_ => hasInstance
				? TryHandleNativeTraitInstanceMethod(invoke, implicitInstance)
				: TryHandleNativeTraitStaticMethod(invoke)
		};
	}

	private ValueInstance ResolveInvokeInstance(InvokeMethodInfo info,
		ValueInstance? implicitInstance) =>
		info.InstanceRegister.HasValue
			? Memory.Registers[info.InstanceRegister.Value]
			: implicitInstance!.Value;

	private bool TryHandleNativeTraitInstanceMethod(Invoke invoke,
		ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var instanceValue = ResolveInvokeInstance(info, implicitInstance);
		if (instanceValue.IsText || instanceValue.IsList || instanceValue.IsDictionary ||
			instanceValue.IsFlatNumeric)
			return false;
		if (!instanceValue.GetType().IsTrait)
			return false;
		var typeInstance = instanceValue.TryGetValueTypeInstance();
		if (typeInstance == null)
			return false;
		var methodIndex = GetTraitDataMethodIndex(instanceValue.GetType(), info.MethodName);
		if (methodIndex < 0 || methodIndex >= typeInstance.Values.Length)
			return false;
		Memory.Registers[invoke.Register] = typeInstance.Values[methodIndex];
		return true;
	}

	private bool TryHandleNativeFileMethod(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		if (info.MethodName == Method.From)
		{
			if (info.ArgumentRegisters.Length != 1)
				return false;
			var pathValue = Memory.Registers[info.ArgumentRegisters[0]];
			if (!pathValue.IsText)
				return false;
			var fileInstance = NativeFileRegistry.Open(executable.basePackage.GetType(Type.File),
				pathValue.Text);
			Memory.Frame.TrackDisposable(fileInstance);
			Memory.Registers[invoke.Register] = fileInstance;
			return true;
		}
		var instance = ResolveInvokeInstance(info, implicitInstance);
		if (!IsFileInstance(instance))
			return false;
		var handle = GetFileHandle(instance);
		switch (info.MethodName)
		{
		case "ReadText":
			Memory.Registers[invoke.Register] = new ValueInstance(NativeFileRegistry.ReadText(handle));
			return true;
		case "ReadBytes":
			Memory.Registers[invoke.Register] = CreateBytesValue(NativeFileRegistry.ReadBytes(handle));
			return true;
		case "Write":
			WriteFile(handle, Memory.Registers[info.ArgumentRegisters[0]]);
			Memory.Registers[invoke.Register] = new ValueInstance(executable.noneType);
			return true;
		case "Delete":
			NativeFileRegistry.Delete(handle);
			Memory.Registers[invoke.Register] = new ValueInstance(executable.noneType);
			return true;
		case "Close":
			NativeFileRegistry.Close(handle);
			Memory.Registers[invoke.Register] = new ValueInstance(executable.noneType);
			return true;
		case "Exists":
			Memory.Registers[invoke.Register] =
				new ValueInstance(executable.booleanType, NativeFileRegistry.Exists(handle));
			return true;
		case "Length":
			Memory.Registers[invoke.Register] =
				new ValueInstance(executable.numberType, NativeFileRegistry.Length(handle));
			return true;
		default:
			return false;
		}
	}

	private long GetFileHandle(ValueInstance instance)
	{
		if (!FileValue.TryGetHandle(instance, executable.basePackage.GetType(Type.File), out var handle))
			throw Fail("File instance has no native handle");
		return handle;
	}

	private void WriteFile(long handle, ValueInstance value)
	{
		if (value.IsText)
			NativeFileRegistry.WriteText(handle, value.Text);
		else if (value.IsList)
			NativeFileRegistry.WriteBytes(handle, FileValue.GetBytes(value));
		else
			throw Fail("File.Write needs Text or Bytes");
	}

	private ValueInstance CreateBytesValue(byte[] bytes)
	{
		var byteType = executable.basePackage.GetType(Type.Byte);
		var bytesType = executable.listType.GetGenericImplementation(byteType);
		return FileValue.CreateBytes(bytesType, byteType, bytes);
	}

	private static int GetTraitDataMethodIndex(Type traitType, string methodName)
	{
		var dataIndex = 0;
		foreach (var method in traitType.Methods)
		{
			if (string.Equals(method.Name, Method.From, StringComparison.OrdinalIgnoreCase))
				continue;
			if (string.Equals(method.Name, methodName, StringComparison.OrdinalIgnoreCase))
				return dataIndex;
			dataIndex++;
		}
		return -1;
	}

	private bool TryHandleNativeTraitStaticMethod(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		var typeName = info.TypeFullName.Split('/').Last();
		var searchDirectory = AppContext.BaseDirectory;
		if (!NativePluginLoader.HasNativeLibrary(typeName, searchDirectory))
			return false;
		if (info.ArgumentRegisters.Length < 4)
			return false;
		var pathArg = Memory.Registers[info.ArgumentRegisters[0]];
		if (!pathArg.IsText)
			return false;
		var colorsArg = Memory.Registers[info.ArgumentRegisters[1]];
		if (!colorsArg.IsList)
			return false;
		var width = (int)Memory.Registers[info.ArgumentRegisters[2]].Number;
		var height = (int)Memory.Registers[info.ArgumentRegisters[3]].Number;
		var pixelData = ExtractRgbaBytes(colorsArg);
		return NativePluginLoader.TrySaveNativeImage(typeName, pathArg.Text, pixelData, width,
			height, searchDirectory);
	}

	private static byte[] ExtractRgbaBytes(ValueInstance listValue)
	{
		var items = listValue.List.Items;
		if (items.Count == 0)
			return [];
		return items[0].TryGetValueTypeInstance() != null
			? ExtractBytesFromColorList(items)
			: ExtractBytesFromNumberList(items);
	}

	private static byte[] ExtractBytesFromNumberList(IReadOnlyList<ValueInstance> items)
	{
		var bytes = new byte[items.Count];
		for (var byteIndex = 0; byteIndex < items.Count; byteIndex++)
			bytes[byteIndex] = (byte)Math.Clamp(items[byteIndex].Number, 0, 255);
		return bytes;
	}

	private static byte[] ExtractBytesFromColorList(IReadOnlyList<ValueInstance> items)
	{
		var bytes = new byte[items.Count * 4];
		var isColorType = IsColorByteType(items[0]);
		for (var colorIndex = 0; colorIndex < items.Count; colorIndex++)
		{
			var typeInst = items[colorIndex].TryGetValueTypeInstance();
			if (typeInst is null || typeInst.Values.Length < 3)
				continue;
			if (isColorType)
			{
				bytes[colorIndex * 4] = ClampToByte(typeInst.Values[0].Number);
				bytes[colorIndex * 4 + 1] = ClampToByte(typeInst.Values[1].Number);
				bytes[colorIndex * 4 + 2] = ClampToByte(typeInst.Values[2].Number);
				bytes[colorIndex * 4 + 3] = typeInst.Values.Length >= 4
					? ClampToByte(typeInst.Values[3].Number)
					: (byte)255;
			}
			else
			{
				bytes[colorIndex * 4] = ClampToByte(typeInst.Values[0].Number * 255);
				bytes[colorIndex * 4 + 1] = ClampToByte(typeInst.Values[1].Number * 255);
				bytes[colorIndex * 4 + 2] = ClampToByte(typeInst.Values[2].Number * 255);
				bytes[colorIndex * 4 + 3] = typeInst.Values.Length >= 4
					? ClampToByte(typeInst.Values[3].Number * 255)
					: (byte)255;
			}
		}
		return bytes;
	}

	private static bool IsColorByteType(ValueInstance colorInstance) =>
		colorInstance.GetType().Name.Equals("Color", StringComparison.OrdinalIgnoreCase);

	private static byte ClampToByte(double value) =>
		(byte)Math.Clamp(Math.Round(value), 0, 255);

	private bool ExecuteFromInvoke(Invoke invoke, Type returnType)
	{
		if (returnType.Name == Type.File)
		{
			var pathValue = Memory.Registers[invoke.MethodInfo.ArgumentRegisters[0]];
			if (!pathValue.IsText)
				return false;
			var fileInstance = NativeFileRegistry.Open(returnType, pathValue.Text);
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
		if (!returnType.IsMutable && (returnType.IsNumber || returnType.IsText ||
			returnType.IsCharacter || returnType.IsEnum || returnType.IsBoolean || returnType.IsNone))
		{
			var info = invoke.MethodInfo;
			Memory.Registers[invoke.Register] = info.ArgumentRegisters.Length > 0
				? Memory.Registers[info.ArgumentRegisters[0]]
				: CreateDefaultValue(returnType);
			return true;
		}
		if (returnType.IsTrait && TryCallNativeFromPlugin(invoke, returnType))
			return true;
		return TryHandleFromConstructor(invoke, returnType);
	}

	private bool TryCallNativeFromPlugin(Invoke invoke, Type returnType)
	{
		var info = invoke.MethodInfo;
		if (info.ArgumentRegisters.Length == 0)
			return false;
		var pathArg = Memory.Registers[info.ArgumentRegisters[0]];
		var pathText = pathArg.IsText
			? pathArg.Text
			: TryExtractTextFromPathType(pathArg);
		if (pathText == null)
			return false;
		var searchDirectory = AppContext.BaseDirectory;
		var bytes = NativePluginLoader.TryLoadNativeLifecycle(returnType.Name, pathText,
			searchDirectory, out var width, out var height);
		if (bytes == null)
			return false;
		var traitValues = BuildNativePluginValues(returnType, bytes, width, height);
		if (traitValues == null)
			return false;
		Memory.Registers[invoke.Register] = new ValueInstance(returnType, traitValues);
		return true;
	}

	private static string? TryExtractTextFromPathType(ValueInstance value)
	{
		var typeInstance = value.TryGetValueTypeInstance();
		if (typeInstance == null || typeInstance.Values.Length == 0)
			return null;
		return typeInstance.Values[0].IsText
			? typeInstance.Values[0].Text
			: null;
	}

	private ValueInstance[]? BuildNativePluginValues(Type traitType, byte[] bytes, int width,
		int height)
	{
		var dataMethods = traitType.Methods
			.Where(m => !string.Equals(m.Name, Method.From, StringComparison.OrdinalIgnoreCase))
			.ToList();
		if (dataMethods.Count == 0)
			return null;
		var values = new ValueInstance[dataMethods.Count];
		for (var methodIndex = 0; methodIndex < dataMethods.Count; methodIndex++)
		{
			var method = dataMethods[methodIndex];
			var returnType = method.ReturnType;
			if (returnType.IsList)
				values[methodIndex] = BuildListValueFromBytes(bytes, returnType);
			else if (returnType.IsNumber)
				values[methodIndex] = new ValueInstance(returnType,
					string.Equals(method.Name, "Width", StringComparison.OrdinalIgnoreCase)
						? width
						: height);
			else
				return null;
		}
		return values;
	}

	private ValueInstance BuildListValueFromBytes(byte[] bytes, Type listType)
	{
		var elementType = listType is GenericTypeImplementation generic
			? generic.ImplementationTypes[0]
			: listType;
		if (elementType.IsNumber || string.Equals(elementType.Name, "Byte",
			StringComparison.OrdinalIgnoreCase))
			return NativePluginLoader.ConvertBytesToValueInstance(bytes, listType);
		var numberType = executable.basePackage.FindType("Number")!;
		var colorCount = bytes.Length / 4;
		var colorValues = new ValueInstance[colorCount];
		for (var colorIndex = 0; colorIndex < colorCount; colorIndex++)
		{
			var r = bytes[colorIndex * 4] / 255.0;
			var g = bytes[colorIndex * 4 + 1] / 255.0;
			var b = bytes[colorIndex * 4 + 2] / 255.0;
			var a = bytes[colorIndex * 4 + 3] / 255.0;
			colorValues[colorIndex] = new ValueInstance(elementType,
				[new ValueInstance(numberType, r), new ValueInstance(numberType, g),
				 new ValueInstance(numberType, b), new ValueInstance(numberType, a)]);
		}
		return new ValueInstance(listType, colorValues);
	}

	private bool TryHandleToConversion(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var conversionType = info.ResolveReturnType(executable.basePackage);
		var rawValue = ResolveInvokeInstance(info, implicitInstance);
		if (conversionType.IsText)
		{
			Memory.Registers[invoke.Register] = ConvertToText(rawValue);
			return true;
		}
		if (conversionType.IsNumber)
		{
			Memory.Registers[invoke.Register] =
				rawValue.IsText
					? new ValueInstance(conversionType, Convert.ToDouble(rawValue.Text))
					: rawValue;
			return true;
		}
		return false;
	}

	private bool TryHandleNativeLength(Invoke invoke, ValueInstance? implicitInstance)
	{
		var instanceValue = ResolveInvokeInstance(invoke.MethodInfo, implicitInstance);
		if (!TryGetNativeLength(instanceValue, invoke.MethodInfo.MethodName, out var lengthValue))
			return false;
		Memory.Registers[invoke.Register] = lengthValue;
		return true;
	}

	private bool TryHandleIncrementDecrement(Invoke invoke, bool isIncrement,
		ValueInstance? implicitInstance)
	{
		var current = ResolveInvokeInstance(invoke.MethodInfo, implicitInstance);
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
			var foundInstructions =
				executable.FindInstructions(strippedTypeName, info.MethodName, info.ParameterNames.Length,
					info.ReturnTypeName) ?? executable.FindInstructions(strictPrefix + strippedTypeName,
					info.MethodName, info.ParameterNames.Length, info.ReturnTypeName);
			if (foundInstructions != null)
				return foundInstructions;
		}
		return null;
	}

	private void InitializeMethodCallScope(InvokeMethodInfo info,
		ValueInstance[] evaluatedArguments, ValueInstance? evaluatedInstance)
	{
		for (var parameterIndex = 0; parameterIndex < info.ParameterNames.Length &&
			parameterIndex < evaluatedArguments.Length; parameterIndex++)
			Memory.Frame.Set(info.ParameterNames[parameterIndex],
				evaluatedArguments[parameterIndex]);
		for (var parameterIndex = evaluatedArguments.Length;
			parameterIndex < info.ParameterNames.Length; parameterIndex++)
			Memory.Frame.Set(info.ParameterNames[parameterIndex],
				new ValueInstance(executable.numberType, 0.0));
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

	private bool TryGetBinaryMembers(Type type, out List<BinaryMember> members)
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
		if (registerStackDepth >= MaxCallDepth)
			throw new StackOverflow(MaxCallDepth);
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
		int SavedInstructionIndex, bool SavedConditionFlag, ValueInstance? SavedReturns,
		CallFrame SavedFrame, int StackDepth, CallFrame Frame);

	private bool TryHandleNativeTextMethod(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var instance = ResolveInvokeInstance(info, implicitInstance);
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
			"LastIndexOf" => new ValueInstance(executable.numberType,
				text.LastIndexOf(args[0].Text, StringComparison.Ordinal)),
			"Substring" => new ValueInstance(
				text.Substring((int)args[0].Number, (int)args[1].Number)),
			"Upper" => new ValueInstance(text.ToUpperInvariant()),
			"Lower" => new ValueInstance(text.ToLowerInvariant()),
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
		var matches = start >= 0 && start + prefix.Length <= text.Length &&
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
			if (IsFileInstance(instance))
			{
				result = new ValueInstance(executable.numberType,
					NativeFileRegistry.Length(GetFileHandle(instance)));
				return true;
			}
		}
		result = default;
		return false;
	}

	private bool IsFileInstance(ValueInstance instance) =>
		instance.GetType().IsSameOrCanBeUsedAs(executable.basePackage.GetType(Type.File));

	private void DisposeTrackedValues(CallFrame frame, ValueInstance? returnValue, CallFrame? parentFrame)
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
			else if (FileValue.TryGetHandle(value, executable.basePackage.GetType(Type.File), out var handle))
				NativeFileRegistry.Close(handle);
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
