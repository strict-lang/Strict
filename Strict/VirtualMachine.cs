using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

//TODO: way too long, remove unused methods, shorten some very badly written long methods, try to convert some switch statements to expression switch .. then split up into 2-3 classes each <400 lines
//nocrunch: no coverage start, performance is very bad when NCrunch is tracking every line
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
		InitializeEntryPointMembers(method);
		return RunInstructions(method.instructions, method.Name);
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
	private static readonly int ValueSymbolId = CallFrame.ValueSymbolId;
	private static readonly int IndexSymbolId = CallFrame.IndexSymbolId;
	private static readonly int OuterSymbolId = CallFrame.OuterSymbolId;
	private static readonly int OuterIndexSymbolId =
		CallFrame.ResolveSymbolId(Type.OuterLowercase + "." + Type.IndexLowercase);
	private static readonly int ElementsSymbolId = CallFrame.ElementsSymbolId;
	private static readonly int CharactersSymbolId = CallFrame.CharactersSymbolId;
	private readonly int noneSymbolId = CallFrame.ResolveSymbolId(Type.None);
	private readonly Dictionary<string, IdentifierAccessPath> identifierAccessPaths =
		new(StringComparer.Ordinal);
	private readonly Dictionary<string, IndexedElementAccessPath> indexedElementAccessPaths =
		new(StringComparer.Ordinal);

	private VirtualMachine RunInstructions(List<Instruction> blockInstructions,
		string context = "body")
	{
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("VirtualMachine.RunInstructions", "context=" + context + ", count=" + blockInstructions.Count);
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

	private void CacheInstructionAccessPaths(IReadOnlyList<Instruction> blockInstructions)
	{
		identifierAccessPaths.EnsureCapacity(identifierAccessPaths.Count + blockInstructions.Count * 2);
		indexedElementAccessPaths.EnsureCapacity(indexedElementAccessPaths.Count + blockInstructions.Count);
		for (var cachedInstructionIndex = 0; cachedInstructionIndex < blockInstructions.Count; cachedInstructionIndex++)
			switch (blockInstructions[cachedInstructionIndex])
			{
			case LoadVariableToRegister loadVariable:
				CacheIdentifierAccessPath(loadVariable.Identifier);
				break;
			case StoreVariableInstruction storeVariable:
				CacheIdentifierAccessPath(storeVariable.Identifier);
				break;
			case StoreFromRegisterInstruction storeFromRegister:
				CacheStoreAccessPath(storeFromRegister.Identifier);
				break;
			case ListCallInstruction listCall:
				CacheIdentifierAccessPath(listCall.Identifier);
				break;
			case WriteToListInstruction writeToList:
				CacheIdentifierAccessPath(writeToList.Identifier);
				break;
			case RemoveInstruction remove:
				CacheIdentifierAccessPath(remove.Identifier);
				break;
			}
	}

	private void CacheStoreAccessPath(string identifier)
	{
		var indexedAccessPath = GetIndexedElementAccessPath(identifier);
		if (!indexedAccessPath.IsValid)
		{
			CacheIdentifierAccessPath(identifier);
			return;
		}
		CacheIdentifierAccessPath(indexedAccessPath.ListPath);
		CacheIdentifierAccessPath(indexedAccessPath.IndexExpression);
	}

	private void CacheIdentifierAccessPath(string identifier)
	{
		if (identifier.Length == 0 || double.TryParse(identifier, out _))
			return;
		_ = GetIdentifierAccessPath(identifier);
	}

	private void InitializeEntryPointMembers(BinaryMethod method)
	{
		//TODO: linq queries are forbidden in inner loops like this!
		var entryType = executable.MethodsPerType.FirstOrDefault(type =>
			type.Value.MethodGroups.Values.Any(overloads => overloads.Contains(method)));
		if (entryType.Value == null)
			return;
		//TODO: too complex, needs to be simplified
		foreach (var member in entryType.Value.Members)
		{
			var value = member.InitialValueExpression is SetInstruction setInstruction
				? CloneConstantValue(setInstruction.ValueInstance)
				: CreateDefaultComplexValue(ResolveBinaryMemberType(member, entryType.Key));
			Memory.Frame.Set(member.Name, value, isMember: true);
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
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("** VirtualMachine.ExecuteInstruction",
				"index=" + instructionIndex + ", type=" + instruction.InstructionType +
				GetInstructionDetails(instruction));
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
			throw new InvalidInstruction(instruction);
		}
	}

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
			Invoke invoke => ", Method=" + DescribeMethodCall(invoke.Method) + ", Register=" +
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

	private static string DescribeMethodCall(MethodCall methodCall)
	{
		var method = methodCall.Method;
		return method.Name + " on " + method.Type.Name + " args=" + methodCall.Arguments.Count +
			", hasInstance=" + (methodCall.Instance != null);
	}

	private static string DescribeValueInstance(ValueInstance value) =>
		!value.HasValue
			? "unset"
			: value.IsText
				? "Text(length=" + value.Text.Length + ")"
				: value.IsList
					? "List(type=" + value.List.ReturnType.Name + ", count=" +
					value.List.Items.Count + ")"
					: value.IsDictionary
						? "Dictionary(count=" + value.GetDictionaryItems().Count + ")"
						: value.TryGetValueTypeInstance() is { } typeInstance
							? "TypeInstance(type=" + typeInstance.ReturnType.Name + ", members=" +
							typeInstance.Values.Length + ")"
							: value.GetType().Name + "(" + value.Number + ")";

	private sealed class InvalidInstruction(Instruction instruction)
		: Exception(instruction.ToString());

	private void ExecuteFieldLoad(FieldLoadInstruction instr)
	{
		var typeInstance = Memory.Registers[instr.ObjectRegister].TryGetValueTypeInstance();
		if (typeInstance == null)
			return;
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
		var values = new ValueInstance[instr.FieldRegisters.Length];
		for (var index = 0; index < instr.FieldRegisters.Length; index++)
			values[index] = Memory.Registers[instr.FieldRegisters[index]];
		Memory.Registers[instr.Register] = new ValueInstance(instr.ReturnType, values);
	}

	private void ExecutePrint(PrintInstruction print)
	{
		if (print.ValueRegister.HasValue)
			Console.WriteLine(print.TextPrefix + Memory.Registers[print.ValueRegister.Value].ToExpressionCodeString());
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
		var variableListElement = Memory.Frame.Get(listCallInstruction.Identifier).List[indexValue];
		Memory.Registers[listCallInstruction.Register] = variableListElement;
	}

	private void ExecuteWriteToList(WriteToListInstruction writeToListInstruction)
	{
		if (!GetIdentifierAccessPath(writeToListInstruction.Identifier).TryResolve(this,
				out var collection) || !collection.IsList)
			throw new InvalidOperationException("Cannot add to non-list variable \"" +
				writeToListInstruction.Identifier + "\"");
		collection.List.Items.Add(Memory.Registers[writeToListInstruction.Register]);
	}

	private void ExecuteWriteToTable(WriteToTableInstruction writeToTableInstruction)
	{
		if (!GetIdentifierAccessPath(writeToTableInstruction.Identifier).TryResolve(this,
				out var collection) || !collection.IsDictionary)
			throw new InvalidOperationException("Cannot add to non-dictionary variable \"" +
				writeToTableInstruction.Identifier + "\"");
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
		while (idx < instructions.Count && instructions[idx].InstructionType != InstructionType.LoopBegin)
			idx++;
		return idx < instructions.Count
			? (LoopBeginInstruction)instructions[idx]
			: throw new InvalidOperationException("No matching LoopBeginInstruction found for LoopEnd");
	}

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
			throw new InvalidOperationException("No precompiled method instructions found for invoke");
		var childScope = InitializeChildScope();
		InitializeMethodCallScope(invoke.Method, evaluatedArgs, evaluatedInstance);
		RunInstructions(invokeInstructions, invoke.Method.Method.Name);
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
			Method.From => instanceExpression == null && (TryHandleAdjustedColorConstructor(invoke) ||
				ExecuteFromInvoke(invoke, methodCall.ReturnType)),
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

	private bool TryHandleAdjustedColorConstructor(Invoke invoke)
	{
		return false;
		/*TODO: this needs to be generalized, this is stupid just for Color
		if (invoke.Method.ReturnType.Name != "Color" || invoke.Method.Arguments.Count != 3 ||
			!TryExtractColorChannelAdjustment(invoke.Method.Arguments[0], "Red", out var listCall,
				out var adjustmentExpression) ||
			!TryExtractMatchingColorChannelAdjustment(invoke.Method.Arguments[1], "Green", listCall,
				adjustmentExpression) ||
			!TryExtractMatchingColorChannelAdjustment(invoke.Method.Arguments[2], "Blue", listCall,
				adjustmentExpression))
			return false;
		var sourceColor = TryResolveIndexedListValue(listCall, out var resolvedColor)
			? resolvedColor
			: EvaluateListCallExpression(listCall);
		var brightness = EvaluateExpression(adjustmentExpression).Number;
		if (!TryGetColorChannels(sourceColor, out var red, out var green, out var blue,
			out var alpha))
			return false;
		Memory.Registers[invoke.Register] = ValueInstance.CreateRgba(invoke.Method.ReturnType,
			red + brightness, green + brightness, blue + brightness,
			alpha ?? GetConstructorMemberValue(invoke.Method.ReturnType, 3).Number);
		return true;
		*/
	}

		/*TODO
	private static bool TryGetColorChannels(ValueInstance colorValue, out double red, out double green,
		out double blue, out double? alpha)
	{
		if (colorValue.TryGetPackedRgbaChannels(out red, out green, out blue, out var packedAlpha))
		{
			alpha = packedAlpha;
			return true;
		}
		var colorTypeInstance = colorValue.TryGetValueTypeInstance();
		if (colorTypeInstance != null && colorTypeInstance.TryGetValue("Red", out var redValue) &&
			colorTypeInstance.TryGetValue("Green", out var greenValue) &&
			colorTypeInstance.TryGetValue("Blue", out var blueValue))
		{
			red = redValue.Number;
			green = greenValue.Number;
			blue = blueValue.Number;
			alpha = colorTypeInstance.TryGetValue("Alpha", out var alphaValue)
				? alphaValue.Number
				: null;
			return true;
		}
		red = green = blue = 0;
		alpha = null;
		return false;
	}

	private static ValueInstance GetConstructorMemberValue(Type targetType, int memberIndex)
	{
		var members = targetType.Members;
		if (memberIndex >= members.Count)
			return default;
		var member = members[memberIndex];
		if (member.Type.IsTrait)
			return CreateTraitInstance(member.Type);
		return member.InitialValue is Value initialValue
			? initialValue.Data
			: CreateDefaultValue(member.Type);
	}

	private static bool TryExtractMatchingColorChannelAdjustment(Expression expression, string expectedMemberName,
		ListCall expectedListCall, Expression expectedAdjustmentExpression) =>
		TryExtractColorChannelAdjustment(expression, expectedMemberName, out var listCall,
			out var adjustmentExpression) && AreEquivalentExpression(listCall.List,
			expectedListCall.List) && AreEquivalentExpression(listCall.Index,
			expectedListCall.Index) && AreEquivalentExpression(adjustmentExpression,
			expectedAdjustmentExpression);

	private static bool TryExtractColorChannelAdjustment(Expression expression,
		string expectedMemberName, out ListCall listCall, out Expression adjustmentExpression)
	{
		if (expression is Binary
			{
				Method.Name: BinaryOperator.Plus,
				Instance: MemberCall
				{
					Member.Name: var memberName,
					Instance: ListCall currentListCall
				},
				Arguments: [var rightExpression]
			} && memberName == expectedMemberName)
		{
			listCall = currentListCall;
			adjustmentExpression = rightExpression;
			return true;
		}
		listCall = null!;
		adjustmentExpression = null!;
		return false;
	}
		*/
	private static bool AreEquivalentExpression(Expression left, Expression right) =>
		ReferenceEquals(left, right) || (left, right) switch
		{
			(Value leftValue, Value rightValue) => leftValue.Data.Equals(rightValue.Data),
			(VariableCall leftVariable, VariableCall rightVariable) => leftVariable.Variable.Name ==
				rightVariable.Variable.Name,
			(ParameterCall leftParameter, ParameterCall rightParameter) =>
				leftParameter.Parameter.Name == rightParameter.Parameter.Name,
			(MemberCall leftMember, MemberCall rightMember) =>
				leftMember.Member.Name == rightMember.Member.Name &&
				(leftMember.Instance == null && rightMember.Instance == null ||
					leftMember.Instance != null && rightMember.Instance != null &&
					AreEquivalentExpression(leftMember.Instance, rightMember.Instance)),
			(ListCall leftListCall, ListCall rightListCall) =>
				AreEquivalentExpression(leftListCall.List, rightListCall.List) &&
				AreEquivalentExpression(leftListCall.Index, rightListCall.Index),
			(Instance, Instance) => true,
			_ => false
		};

	//TODO: find all [.. with existing list and no changes, all those cases need to be removed, there is a crazy amount of those added (54 wtf)!
	private List<Instruction>? GetPrecompiledMethodInstructions(Method method) =>
		executable.FindInstructions(method.Type, method) ??
		executable.FindInstructions(method.Type.Name, method.Name,
			method.Parameters.Count, method.ReturnType.Name) ??
		executable.FindInstructions(
			nameof(Strict) + Context.ParentSeparator + method.Type.Name, method.Name,
			method.Parameters.Count, method.ReturnType.Name) ??
		FindInstructionsWithStrippedPackagePrefix(method);

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

	/// <summary>
	/// Handles From constructor calls like SimpleCalculator(2, 3) by creating a ValueInstance
	/// with evaluated argument values for each non-trait member.
	/// </summary>
	//TODO: remove
	private bool TryHandleFromConstructor(Invoke invoke)
	{
		if (invoke.Method.Method.Name != Method.From || invoke.Method.Instance != null)
			return false;
		return TryHandleFromConstructor(invoke, invoke.Method.ReturnType);
	}

	private bool TryHandleFromConstructor(Invoke invoke, Type targetType)
	{
		if (targetType is GenericTypeImplementation)
			return false;
		var members = targetType.Members;
		var hasBinaryMembers = TryGetBinaryMembers(targetType, out var binaryMembers);
		if (members.Count == 0 && hasBinaryMembers)
		{
			//TODO: called almost 1 million times? wtf?
			Memory.Registers[invoke.Register] = new ValueInstance(targetType,
				CreateConstructorValuesFromBinaryMembers(targetType, invoke, binaryMembers));
			return true;
		}
		var values = new ValueInstance[members.Count];
		for (var parameterIndex = 0; parameterIndex < invoke.Method.Method.Parameters.Count; parameterIndex++)
		{
			//wtf, another 1.2m calls?
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
				values[memberIndex] = CreateDefaultComplexValue(memberType);
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
			if (!values[memberIndex].IsList || values[memberIndex].List.Count > 0 ||
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
			var defaultElement = CreateDefaultComplexValue(elementType);
			values[memberIndex] = new ValueInstance(members[memberIndex].Type, defaultElement,
				length.Value);
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
						: memberType.IsNone
							? new ValueInstance(memberType)
							: memberType.Members.Count > 0 && !memberType.IsMutable
								? new ValueInstance(memberType)
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
				var childScope = InitializeChildScope();
				Memory.Frame.Set(Type.ValueLowercase, memberValue, isMember: true);
				TrySetScopeMembersFromTypeMembers(typeInstance!);
				RunInstructions(methodInstructions);
				var result = Returns;
				CleanupChildScope(childScope);
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
	//TODO: this is called 4 million times, stupid!
	private ValueInstance EvaluateExpression(Expression expression)
	{
		if (expression is Value value)
			return value.Data;
		if (expression is VariableCall variableCall && variableCall.IsConstant &&
			variableCall.Variable.InitialValue is Value constantValue)
			return constantValue.Data;
		if (expression is VariableCall directVariable)
			return EvaluateVariableCall(directVariable);
		if (expression is ParameterCall parameterCall)
			return Memory.Frame.Get(CallFrame.ResolveSymbolId(parameterCall.Parameter.Name));
		if (expression is Instance)
			return Memory.Frame.Get(ValueSymbolId);
		if (expression is MemberCall memberCall)
			return EvaluateMemberCall(memberCall);
		if (expression is Binary binary)
			return EvaluateBinary(binary); //TODO: eats up 75% of the time here! 0.7m calls
		if (expression is MethodCall methodCall)
			return EvaluateMethodCall(methodCall);
		if (expression is ListCall listCall)
			return EvaluateListCallExpression(listCall); //TODO: another almost 25% here, 0.23m calls
		return TryResolveExpressionFallback(expression, out var resolvedValue)
			? resolvedValue
			: throw new InvalidOperationException("Could not evaluate expression " + expression);
	}

	private ValueInstance EvaluateVariableCall(VariableCall variableCall) =>
		Memory.Frame.Get(variableCall.Variable.Name == Type.OuterLowercase
			? OuterSymbolId
			: CallFrame.ResolveSymbolId(variableCall.Variable.Name));

	private ValueInstance EvaluateListCallExpression(ListCall listCall)
	{
		if (TryResolveIndexedListValue(listCall, out var directListElement))
			return directListElement;
		if (TryEvaluateDirectSpecialListCall(listCall, out var directValue))
			return directValue;
		var listValue = EvaluateExpression(listCall.List);
		var indexValue = TryEvaluateDirectIndexValue(listCall.Index, out var directIndexValue)
			? directIndexValue
			: EvaluateExpression(listCall.Index);
		var index = (int)indexValue.Number;
		if (listValue.IsList || listValue.IsText ||
			listValue.TryGetValueTypeInstance()?.ReturnType.IsList == true)
			return listValue.GetIteratorValue(executable.characterType, index);
		if (listValue.TryGetValueTypeInstance() is { } typeInstance)
		{
			if (typeInstance.TryGetValue(Type.ElementsLowercase, out var elementsValue) &&
				(elementsValue.IsList || elementsValue.IsText))
				return elementsValue.GetIteratorValue(executable.characterType, index);
			for (var valueIndex = 0; valueIndex < typeInstance.Values.Length; valueIndex++)
				if (typeInstance.Values[valueIndex].IsText)
					return typeInstance.Values[valueIndex].
						GetIteratorValue(executable.characterType, index);
		}
		return TryResolveExpressionFallback(listCall, out var resolvedValue)
			? resolvedValue
			: throw new InvalidOperationException("Could not evaluate list call " + listCall);
	}

	private bool TryResolveIndexedListValue(ListCall listCall, out ValueInstance value)
	{
		value = default;
		if (!TryEvaluateDirectIndexValue(listCall.Index, out var indexValue))
			return false;
		var listValue = listCall.List switch
		{
			VariableCall variableCall => EvaluateVariableCall(variableCall),
			MemberCall memberCall => EvaluateMemberCall(memberCall),
			_ => default
		};
		if (!listValue.HasValue)
			return false;
		var index = (int)indexValue.Number;
		if (listValue.IsList && index >= 0 && index < listValue.List.Count)
		{
			value = listValue.List[index];
			return true;
		}
		if (listValue.TryGetValueTypeInstance() is { } typeInstance &&
			typeInstance.TryGetValue(Type.ElementsLowercase, out var elementsValue) &&
			elementsValue.IsList && index >= 0 && index < elementsValue.List.Count)
		{
			value = elementsValue.List[index];
			return true;
		}
		return false;
	}

	private bool TryEvaluateDirectSpecialListCall(ListCall listCall, out ValueInstance value)
	{
		if (TryEvaluateDirectIndexValue(listCall.Index, out var indexValue))
		{
			var directListValue = listCall.List switch
			{
				VariableCall { Variable.Name: Type.ValueLowercase } => Memory.Frame.Get(ValueSymbolId),
				VariableCall { Variable.Name: Type.OuterLowercase } => Memory.Frame.Get(OuterSymbolId),
				MemberCall { Instance: VariableCall { Variable.Name: Type.ValueLowercase } } memberCall =>
					EvaluateMemberCall(memberCall),
				MemberCall { Instance: VariableCall { Variable.Name: Type.OuterLowercase } } memberCall =>
					EvaluateMemberCall(memberCall),
				_ => default
			};
			if (directListValue.HasValue)
			{
				value = directListValue.GetIteratorValue(executable.characterType,
					(int)indexValue.Number);
				return true;
			}
		}
		value = default;
		return false;
	}

	private bool TryEvaluateDirectIndexValue(Expression indexExpression, out ValueInstance value)
	{
		switch (indexExpression)
		{
		case Value constantValue:
			value = constantValue.Data;
			return true;
		case VariableCall { Variable.Name: Type.IndexLowercase }:
			return Memory.Frame.TryGet(IndexSymbolId, out value);
		case VariableCall variableCall when variableCall.IsConstant &&
			variableCall.Variable.InitialValue is Value constantVariableValue:
			value = constantVariableValue.Data;
			return true;
		default:
			value = default;
			return false;
		}
	}

	private ValueInstance EvaluateMemberCall(MemberCall memberCall)
	{
		if (memberCall.Instance == null)
		{
			if (memberCall.Member.InitialValue is Value enumValue)
				return enumValue.Data;
			return TryGetFrameValue(CallFrame.ResolveSymbolId(memberCall.Member.Name),
				out var scopedMemberValue)
				? scopedMemberValue
				: throw new InvalidOperationException("Could not resolve member " + memberCall.Member.Name);
		}
		var instanceValue = EvaluateExpression(memberCall.Instance);
		if (TryGetNativeLength(instanceValue, memberCall.Member.Name, out var lengthValue))
			return lengthValue;
		/*TODO
		if (instanceValue.TryGetPackedRgbaMember(memberCall.Member.Name, out var packedMemberValue))
			return packedMemberValue;
			*/
		if (instanceValue.TryGetFlatNumericMember(memberCall.Member.Name, out var flatMemberValue))
			return flatMemberValue;
		var typeInstance = instanceValue.TryGetValueTypeInstance();
		if (typeInstance != null &&
			typeInstance.TryGetValue(memberCall.Member.Name, out var memberValue))
			return memberValue;
		if (instanceValue.IsText && memberCall.Member.Name is "characters" or "elements")
			return instanceValue;
		return TryResolveExpressionFallback(memberCall, out var resolvedValue)
			? resolvedValue
			: throw new InvalidOperationException("Could not evaluate member call " + memberCall);
	}

	private bool TryResolveExpressionFallback(Expression expression, out ValueInstance value)
	{
		switch (expression)
		{
		case MemberCall memberCall when memberCall.Instance == null:
			return TryGetFrameValue(CallFrame.ResolveSymbolId(memberCall.Member.Name), out value);
		case MemberCall memberCall:
			return GetIdentifierAccessPath(memberCall.ToString()).TryResolve(this, out value);
		case ListCall listCall:
			return GetIdentifierAccessPath(listCall.ToString()).TryResolve(this, out value);
		case VariableCall variableCall:
			return TryGetFrameValue(CallFrame.ResolveSymbolId(variableCall.Variable.Name), out value);
		case ParameterCall parameterCall:
			return TryGetFrameValue(CallFrame.ResolveSymbolId(parameterCall.Parameter.Name), out value);
		case Instance:
			return TryGetFrameValue(ValueSymbolId, out value);
		default:
			value = default;
			return false;
		}
	}

	//TODO: cumbersome, simplify in a few lines
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
			var childScope = InitializeChildScope();
			InitializeMethodCallScope(call, evaluatedArguments, evaluatedInstance);
			RunInstructions(precompiledInstructions);
			var precompiledResult = Returns;
			CleanupChildScope(childScope);
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
				var flatNumericInstance = evaluatedInstance.Value.TryGetFlatNumericArrayInstance();
				if (flatNumericInstance != null)
				{
					var flatMembers = flatNumericInstance.ReturnType.Members;
					for (var memberIndex = 0; memberIndex < flatMembers.Count &&
						memberIndex < flatNumericInstance.FlatWidth; memberIndex++)
						if (!flatMembers[memberIndex].Type.IsTrait)
							Memory.Frame.Set(flatMembers[memberIndex].Name,
								new ValueInstance(flatMembers[memberIndex].Type,
									flatNumericInstance.GetFlat(memberIndex)), isMember: true);
				}
				else
				{
					var typeInstance = evaluatedInstance.Value.TryGetValueTypeInstance();
					if (typeInstance != null)
						TrySetScopeMembersFromTypeMembers(typeInstance);
				}
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
		if (TryCreateCurrentAdjustBrightnessDefaultColorImage(targetType, out var colorImage))
			return colorImage;
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

	private bool TryCreateCurrentAdjustBrightnessDefaultColorImage(Type targetType,
		out ValueInstance colorImage)
	{
		if (targetType.Name != "ColorImage")
		{
			colorImage = default;
			return false;
		}
		var frame = Memory.Frame;
		if (!frame.TryGet(CallFrame.ResolveSymbolId("width"), out var width) ||
			!frame.TryGet(CallFrame.ResolveSymbolId("height"), out var height))
		{
			colorImage = default;
			return false;
		}
		var members = targetType.Members;
		if (members.Count < 2)
		{
			colorImage = default;
			return false;
		}
		var sizeValue = new ValueInstance(members[0].Type, [width, height]);
		var colorType = executable.basePackage.FindType("Color");
		if (colorType == null)
		{
			colorImage = default;
			return false;
		}
		var defaultColor = new ValueInstance(colorType, [
			new ValueInstance(executable.numberType, 0),
			new ValueInstance(executable.numberType, 0),
			new ValueInstance(executable.numberType, 0),
			new ValueInstance(executable.numberType, 1)
		]);
		var colorCount = (int)(width.Number * height.Number);
		var colors = new ValueInstance[colorCount];
		for (var colorIndex = 0; colorIndex < colorCount; colorIndex++)
			colors[colorIndex] = defaultColor;
		var colorList = new ValueInstance(members[1].Type, colors);
		colorImage = new ValueInstance(targetType, [sizeValue, colorList]);
		return true;
	}

	private bool GetValueByKeyForDictionaryAndStoreInRegister(Invoke invoke)
	{
		if (invoke.Method.Method.Name != "Get" ||
			invoke.Method.Instance?.ReturnType.IsDictionary != true)
			return false;
		var keyArg = invoke.Method.Arguments[0];
		var keyData = keyArg is Value argValue
			? argValue.Data
			: EvaluateExpression(keyArg);
		var dictionary = EvaluateExpression(invoke.Method.Instance);
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
				throw new InvalidOperationException("Could not resolve variable " +
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

	private IdentifierAccessPath GetIdentifierAccessPath(string identifier) =>
		identifierAccessPaths.TryGetValue(identifier, out var accessPath)
			? accessPath
			: identifierAccessPaths[identifier] = IdentifierAccessPath.Parse(identifier);

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
		if (!TrySetTypeMemberValue(typeInstance, memberName, value))
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
			_ => Memory.Registers[instruction.Registers[^1]]
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
			throw new NotSupportedException("Texts cannot be subtracted: " + left + " - " + right);
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
			if (instructions[index].InstructionType == InstructionType.JumpEnd &&
				((JumpToId)instructions[index]).Id == id)
				return index;
		return -1;
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

	private static bool TrySetTypeMemberValue(ValueTypeInstance typeInstance, string memberName,
		ValueInstance value) =>
		typeInstance.TrySetValue(memberName, value);

	private readonly record struct IdentifierAccessPath(int RootSymbolId, string[] MemberNames,
		bool IsNone)
	{
		public bool TryResolve(VirtualMachine vm, out ValueInstance value)
		{
			if (IsNone)
			{
				value = default;
				return true;
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
				return new IdentifierAccessPath(-1, [], true);
			var firstDotIndex = identifier.IndexOf('.');
			if (firstDotIndex < 0)
				return new IdentifierAccessPath(CallFrame.ResolveSymbolId(identifier), [], false);
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
			return new IdentifierAccessPath(rootSymbolId, memberNames, false);
		}

		public IdentifierAccessPath GetParentPath() =>
			MemberNames.Length == 1
				? new IdentifierAccessPath(RootSymbolId, [], false)
				: new IdentifierAccessPath(RootSymbolId, MemberNames[..^1], false);
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