using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

/// <summary>
/// Converts an expression into a <see cref="BinaryExecutable"/>, mostly from calling the Run
/// method of a .strict type, but can be any expression. Will get all used types with their
/// members and used methods recursively, execution and serialization can be done independently.
/// </summary>
public sealed class BinaryGenerator
{
	public BinaryGenerator(MethodCall methodCall)
	{
		entryMethodCall = methodCall;
		entryTypeFullName = methodCall.Method.Type.FullName;
		if (methodCall.Instance is MethodCall instanceCall)
			AddInstanceMemberVariables(instanceCall);
		AddMethodParameterVariables(methodCall);
		//TODO: this randomly crashes VirtualMachineTests.Enum stuff .. bad anyway
		var methodBody = methodCall.Method.GetBodyAndParseIfNeeded();
		Expressions = methodBody is Body body
			? body.Expressions
			: [methodBody];
		ReturnType = methodCall.Method.ReturnType;
		binary = new BinaryExecutable(GetBasePackage(methodCall));
	}

	private BinaryGenerator(Package basePackage, IReadOnlyList<Expression> expressions, Type returnType)
	{
		binary = new BinaryExecutable(basePackage);
		Expressions = expressions;
		ReturnType = returnType;
		entryTypeFullName = "";
	}

	private readonly BinaryExecutable binary;
	private readonly MethodCall? entryMethodCall;
	private readonly string entryTypeFullName;
	private readonly List<Instruction> instructions = []; //TODO: why not keep this in BinaryMethod
	private readonly Dictionary<string, Type> dependencyTypes = new(StringComparer.Ordinal);
	private readonly Registry registry = new();
	private readonly Stack<int> idStack = new();
	private readonly Register[] registers = Enum.GetValues<Register>();
	private readonly List<Method> discoveredInvokeMethods = [];
	internal IReadOnlyList<Method> DiscoveredInvokeMethods => discoveredInvokeMethods;
	private IReadOnlyList<Expression> Expressions { get; } //TODO: stupid, remove
	private Type ReturnType { get; } //TODO: stupid, remove
	private int conditionalId; //TODO: a bit strange
	private int forResultId;
	private int listResultId;

	private void AddInstruction(Instruction instruction, int sourceLine)
	{
		instruction.SourceLine = sourceLine;
		instructions.Add(instruction);
	}

	public BinaryExecutable Generate() =>
		entryMethodCall is { Method.Name: Method.Run, Arguments.Count: 0 }
			? Generate(entryMethodCall.Method,
				entryMethodCall.Method.Type.Methods.Where(method => method.Name == Method.Run).ToArray())
			: Generate(entryTypeFullName, Expressions, ReturnType);

	//TODO: this is convoluted and not good
	public static BinaryExecutable GenerateFromRunMethods(Method preferredEntryMethod,
		IReadOnlyList<Method> runMethods)
	{
		var generator = new BinaryGenerator(GetBasePackage(preferredEntryMethod), [],
			preferredEntryMethod.ReturnType);
		return generator.Generate(preferredEntryMethod, runMethods);
	}

	public static List<Instruction> GenerateInlineInstructions(Package basePackage,
		Expression expression) =>
		new BinaryGenerator(basePackage, [expression], expression.ReturnType).GenerateInstructionList();

	//TODO: unused again?
	public BinaryExecutable Generate(string typeFullName, Expression entryPointExpression) =>
		Generate(typeFullName, [entryPointExpression], entryPointExpression.ReturnType);

	private BinaryExecutable Generate(Method preferredEntryMethod, IReadOnlyList<Method> runMethods)
	{
		var methodsByType = GenerateRunMethods(runMethods, preferredEntryMethod.Type);
		AddGeneratedTypes(methodsByType, preferredEntryMethod.Type);
		binary.SetEntryPoint(GetBinaryTypeName(preferredEntryMethod.Type, preferredEntryMethod.Type),
			preferredEntryMethod.Name, preferredEntryMethod.Parameters.Count,
			GetBinaryTypeName(preferredEntryMethod.ReturnType, preferredEntryMethod.Type));
		return binary;
	}

	private BinaryExecutable Generate(string typeFullName,
		IReadOnlyList<Expression> entryExpressions, Type runReturnType)
	{
		var methodsByType = CompileMethodsFromExpressions(typeFullName, entryExpressions, runReturnType);
		var entryType = FindEntryType(typeFullName);
		if (entryType == null)
			foreach (var (compiledTypeFullName, methodGroups) in methodsByType)
				binary.AddType(compiledTypeFullName, [], methodGroups);
		else
		{
			CollectTypeDependency(entryType, true);
			CollectTypeDependency(runReturnType, false);
			foreach (var expression in entryExpressions)
				CollectExpressionDependencies(expression);
			AddGeneratedTypes(methodsByType, entryType);
		}
		return binary;
	}

	private Type? FindEntryType(string typeFullName) =>
		string.IsNullOrEmpty(typeFullName)
			? null
			: binary.basePackage.FindFullType(typeFullName) ?? binary.basePackage.FindType(typeFullName);

	private static Package GetBasePackage(Expression expression)
	{
		Context context = expression.ReturnType;
		while (context is not Package)
			context = context.Parent;
		return (Package)context;
	}

	private static Package GetBasePackage(Method method)
	{
		Context context = method.Type;
		while (context is not Package)
			context = context.Parent;
		return (Package)context;
	}

	//TODO: unused again?
	private static string GetEntryTypeFullName(Expression expression) =>
		expression is MethodCall methodCall
			? methodCall.Method.Type.FullName
			: expression.ReturnType.FullName;

	//TODO: unused again?
	private void AddMembersFromCaller(ValueInstance instance)
	{
		instructions.Add(new StoreVariableInstruction(instance, Type.ValueLowercase, isMember: true));
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null)
		{
			var members = typeInstance.ReturnType.Members;
			for (var memberIndex = 0; memberIndex < members.Count && memberIndex < typeInstance.Values.Length;
				memberIndex++)
				if (!members[memberIndex].Type.IsTrait)
					instructions.Add(new StoreVariableInstruction(typeInstance.Values[memberIndex],
						members[memberIndex].Name, isMember: true));
			return;
		}
		var firstNonTraitMember = instance.GetType().Members.
			FirstOrDefault(member => !member.Type.IsTrait);
		if (firstNonTraitMember != null)
			instructions.Add(new StoreVariableInstruction(instance,
				firstNonTraitMember.Name, isMember: true));
	}

	private static ValueInstance GetValueInstanceFromExpression(Expression expression) =>
		expression switch
		{
			List list => list.TryGetConstantData() ?? throw new NotSupportedException(
				"Dynamic lists (mutable or containing any non constant expression) are not supported yet"),
			Value val => val.Data,
			MemberCall memberCall when memberCall.Member.InitialValue != null =>
				memberCall.Member.InitialValue is Value enumValue
					? enumValue.Data
					: new ValueInstance(memberCall.Member.InitialValue.ToString()),
			_ => new ValueInstance(expression.ToString()) //ncrunch: no coverage
		};

	private void AddInstanceMemberVariables(MethodCall instance)
	{
		for (var parameterIndex = 0; parameterIndex < instance.Method.Parameters.Count;
			parameterIndex++)
		{
			var parameter = instance.Method.Parameters[parameterIndex];
			var member = instance.ReturnType.Members.FirstOrDefault(typeMember =>
				typeMember.Name.Equals(parameter.Name, StringComparison.OrdinalIgnoreCase));
			if (member == null)
				continue;
			var argumentExpression = parameterIndex < instance.Arguments.Count
				? instance.Arguments[parameterIndex]
				: parameter.DefaultValue ?? member.InitialValue;
			if (argumentExpression == null)
				continue;
			if (parameter.Type.IsList && instance.Arguments is not [List])
			{
				var listItems = instance.Arguments.Select(GetValueInstanceFromExpression).ToArray();
				instructions.Add(new StoreVariableInstruction(
					new ValueInstance(parameter.Type, listItems), member.Name, isMember: true));
			}
			else
				instructions.Add(new StoreVariableInstruction(
					GetValueInstanceFromExpression(argumentExpression), member.Name, isMember: true));
		}
	}

	private void AddMethodParameterVariables(MethodCall methodCall)
	{
		for (var index = 0; index < methodCall.Method.Parameters.Count &&
			index < methodCall.Arguments.Count; index++)
			StoreEntryVariable(methodCall.Method.Parameters[index].Name, methodCall.Arguments[index]);
	}

	private void StoreEntryVariable(string identifier, Expression expression)
	{
		if (TryGetConstantValueInstance(expression, out var value))
			instructions.Add(new StoreVariableInstruction(value, identifier));
		else
		{
			GenerateInstructionFromExpression(expression);
			instructions.Add(new StoreFromRegisterInstruction(registry.PreviousRegister, identifier));
		}
	}

	private static bool TryGetConstantValueInstance(Expression expression, out ValueInstance value)
	{
		switch (expression)
		{
		case List list:
			var listValue = list.TryGetConstantData();
			if (listValue != null)
			{
				value = listValue.Value;
				return true;
			}
			break;
		case Value constantValue:
			value = constantValue.Data;
			return true;
		case MemberCall memberCall when memberCall.Member.InitialValue != null:
			value = memberCall.Member.InitialValue is Value enumValue
				? enumValue.Data
				: new ValueInstance(memberCall.Member.InitialValue.ToString());
			return true;
		}
		value = default;
		return false;
	}

	private void GenerateListExpression(List list)
	{
		if (list.TryGetConstantData() is { } constantList)
		{
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(), constantList));
			return;
		}
		var listVariable = $"listResult{listResultId++}";
		instructions.Add(new StoreVariableInstruction(new ValueInstance(list.ReturnType,
			Array.Empty<ValueInstance>()), listVariable));
		for (var valueIndex = 0; valueIndex < list.Values.Count; valueIndex++)
		{
			GenerateInstructionFromExpression(list.Values[valueIndex]);
			instructions.Add(new WriteToListInstruction(registry.PreviousRegister, listVariable));
		}
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), listVariable));
	}

	private List<Instruction> GenerateInstructions(IReadOnlyList<Expression> expressions)
	{
		for (var i = 0; i < expressions.Count; i++)
			if ((ReferenceEquals(expressions[i], Expressions[^1]) ||
					expressions[i] is Expressions.Return) && expressions[i] is not If &&
				expressions[i] is not SelectorIf)
				GenerateReturnInstruction(expressions[i]);
			else
				GenerateInstructionFromExpression(expressions[i]);
		return instructions;
	}

	private void GenerateReturnInstruction(Expression expression)
	{
		if (expression is Return returnExpression)
			expression = returnExpression.Value;
		if (TryGenerateNumberForLoopReturn(expression))
			return;
		if (TryGenerateListForLoopReturn(expression))
			return;
		GenerateInstructionFromExpression(expression);
		instructions.Add(new ReturnInstruction(registry.PreviousRegister));
	}

	private bool TryGenerateNumberForLoopReturn(Expression expression)
	{
		if (expression is not For forExpression || !ReturnType.IsNumber)
			return false;
		GenerateInstructionForNumberAggregation(forExpression);
		return true;
	}

	private bool TryGenerateListForLoopReturn(Expression expression)
	{
		if (expression is not For forExpression || !ShouldAggregateLoopToList())
			return false;
		GenerateInstructionForListAggregation(forExpression);
		instructions.Add(new ReturnInstruction(registry.PreviousRegister));
		return true;
	}

	private void GenerateInstructionForNumberAggregation(For forExpression)
	{
		var resultVariable = $"forResult{forResultId++}";
		instructions.Add(new StoreVariableInstruction(new ValueInstance(ReturnType, 0), resultVariable));
		GenerateLoopInstructions(forExpression, resultVariable, LoopAggregation.Number);
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), resultVariable));
		instructions.Add(new ReturnInstruction(registry.PreviousRegister));
	}

	private void GenerateInstructionForListAggregation(For forExpression)
	{
		var resultVariable = $"forResult{forResultId++}";
		var listType = GetListType(forExpression.Body.ReturnType);
		//TODO: why does this create a new ValueInstance, no good, especially the array version!
		instructions.Add(new StoreVariableInstruction(new ValueInstance(listType,
			Array.Empty<ValueInstance>()), resultVariable));
		GenerateLoopInstructions(forExpression, resultVariable, LoopAggregation.List);
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), resultVariable));
	}

	private bool ShouldAggregateLoopToList() =>
		ReturnType.IsIterator || ReturnType.IsList;

	private Type GetListType(Type elementType) =>
		binary.basePackage.FindType(Type.List)?.GetGenericImplementation(elementType) ??
		throw new InvalidOperationException("List type not found for loop aggregation");

	private enum LoopAggregation
	{
		None,
		Number,
		List
	}

	//TODO: try optimize into a expression switch
	private void GenerateInstructionFromExpression(Expression expression)
	{
		var countBefore = instructions.Count;
		switch (expression)
		{
		case Body body:
			GenerateInstructions(body.Expressions);
			return;
		case Binary binaryExpression:
			if (binaryExpression.Method.Name == BinaryOperator.Is)
				return;
			if (!CanGenerateDirectBinaryInstruction(binaryExpression.Method.Name))
			{
				GenerateMethodCallInstruction(binaryExpression);
				break;
			}
			GenerateCodeForBinary(binaryExpression);
			break;
		case If ifExpression:
			GenerateIfInstructions(ifExpression);
			return;
		case SelectorIf selectorIf:
			GenerateSelectorIfInstructions(selectorIf);
			return;
		case Declaration { IsMutable: true } mutableDeclaration:
			GenerateForAssignmentOrDeclaration(mutableDeclaration.Value, mutableDeclaration.Name);
			return;
		case Declaration declaration:
			GenerateForAssignmentOrDeclaration(declaration.Value, declaration.Name);
			return;
		case For forExpression:
			GenerateLoopInstructions(forExpression);
			return;
		case MutableReassignment reassignment:
			GenerateForAssignmentOrDeclaration(reassignment.Value, reassignment.Name);
			return;
		case MemberCall memberCall:
			GenerateMemberCallInstruction(memberCall);
			break;
		case VariableCall:
		case ParameterCall:
		case Instance:
			instructions.Add(
				new LoadVariableToRegister(registry.AllocateRegister(), expression.ToString()));
			break;
		case List list:
			GenerateListExpression(list);
			break;
		case Value value:
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
				GetValueInstanceFromExpression(value)));
			break;
		case MethodCall methodCall:
			GenerateMethodCallInstruction(methodCall);
			break;
		case ListCall listCall:
			GenerateInstructionFromExpression(listCall.Index);
			var indexRegister = registry.PreviousRegister;
			instructions.Add(new ListCallInstruction(registry.AllocateRegister(), indexRegister,
				listCall.List.ToString()));
			break;
		default:
			throw new NotSupportedException(expression.ToString()); //ncrunch: no coverage
		}
		var sourceLine = expression.LineNumber;
		for (var instructionIndex = countBefore; instructionIndex < instructions.Count; instructionIndex++)
			if (instructions[instructionIndex].SourceLine == 0)
				instructions[instructionIndex].SourceLine = sourceLine;
	}

	private void GenerateMemberCallInstruction(MemberCall memberCall)
	{
		if (memberCall.Member.InitialValue != null && memberCall.Member.DefinedIn.IsEnum)
		{
			TryGenerateForEnum(memberCall.Member.DefinedIn, memberCall.Member.InitialValue);
			return;
		}
		if (memberCall.Instance == null)
			instructions.Add(
				new LoadVariableToRegister(registry.AllocateRegister(), memberCall.ToString()));
		//TODO: no tests exist for any of this yet:
		else if (memberCall.Member.InitialValue != null)
			TryGenerateForEnum(memberCall.Member.DefinedIn, memberCall.Member.InitialValue);
		else
			instructions.Add(
				new LoadVariableToRegister(registry.AllocateRegister(), memberCall.ToString()));
	}

	private void GenerateMethodCallInstruction(MethodCall methodCall)
	{
		if (TryGenerateInstructionForCollectionManipulation(methodCall))
			return;
		if (TryGeneratePrintInstruction(methodCall))
			return;
		if (methodCall.Method.Name != Method.From)
			discoveredInvokeMethods.Add(methodCall.Method);
		Register? instanceRegister = null;
		if (methodCall.Instance != null)
		{
			GenerateInstructionFromExpression(methodCall.Instance);
			instanceRegister = registry.PreviousRegister;
		}
		var argumentRegisters = new Register[methodCall.Arguments.Count];
		for (var argumentIndex = 0; argumentIndex < methodCall.Arguments.Count; argumentIndex++)
		{
			GenerateInstructionFromExpression(methodCall.Arguments[argumentIndex]);
			argumentRegisters[argumentIndex] = registry.PreviousRegister;
		}
		var parameterNames = new string[methodCall.Method.Parameters.Count];
		for (var paramIndex = 0; paramIndex < methodCall.Method.Parameters.Count; paramIndex++)
			parameterNames[paramIndex] = methodCall.Method.Parameters[paramIndex].Name;
		var methodInfo = new InvokeMethodInfo(
			methodCall.Method.Type.FullName,
			methodCall.Method.Name, parameterNames,
			GetBinaryTypeName(methodCall.ReturnType, methodCall.Method.Type),
			argumentRegisters, instanceRegister);
		instructions.Add(new Invoke(registry.AllocateRegister(), methodInfo));
	}

	private void TryGenerateForEnum(Type type, Expression value)
	{
		if (type.IsEnum)
		{
			var data = value is Value val
				? val.Data
				: new ValueInstance(value.ToString());
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(), data));
		}
	}

	private bool TryGeneratePrintInstruction(MethodCall methodCall)
	{
		if (methodCall.Instance is not MemberCall memberCall)
			return false;
		if (memberCall.Member.Type.Name is not (Type.Logger or Type.TextWriter or Type.System))
			return false;
		if (methodCall.Arguments.Count == 0)
		{
			instructions.Add(new PrintInstruction(""));
			return true;
		}
		var argument = methodCall.Arguments[0];
		if (argument is Value textValue && textValue.Data.IsText)
		{
			instructions.Add(new PrintInstruction(textValue.Data.Text));
			return true;
		}
		if (argument is Binary binaryExpression)
		{
			var prefix = ExtractTextPrefix(binaryExpression.Instance);
			var valueExpression = UnwrapToConversion(binaryExpression.Arguments[0]);
			GenerateInstructionFromExpression(valueExpression);
			instructions.Add(new PrintInstruction(prefix, registry.PreviousRegister,
				valueExpression.ReturnType.IsText));
			return true;
		}
		if (argument is MethodCall argumentMethodCall)
		{
			GenerateInstructionFromExpression(argumentMethodCall);
			instructions.Add(new PrintInstruction("", registry.PreviousRegister,
				argumentMethodCall.ReturnType.IsText));
			return true;
		}
		instructions.Add(new PrintInstruction(argument.ToString()));
		return true;
	}

	private void GenerateLoopInstructions(For forExpression, string? aggregationTarget = null,
		LoopAggregation aggregation = LoopAggregation.None)
	{
		var forSourceLine = forExpression.LineNumber;
		var instructionCountBeforeLoopStart = instructions.Count;
		var customVariableNames = forExpression.CustomVariables.Select(variable =>
			variable.ToString()).ToArray();
		var iterator = GetLoopIteratorExpression(forExpression.Iterator);
		LoopBeginInstruction loopBegin;
		if (iterator is MethodCall rangeExpression &&
			iterator.ReturnType.Name == Type.Range &&
			rangeExpression.Method.Name == Method.From)
		{
			loopBegin = GenerateInstructionForRangeLoopInstruction(rangeExpression,
				customVariableNames);
			loopBegin.SourceLine = forSourceLine;
		}
		else
		{
			var iteratorStart = instructions.Count;
			GenerateInstructionFromExpression(iterator);
			for (var instructionIndex = iteratorStart; instructionIndex < instructions.Count;
				instructionIndex++)
				instructions[instructionIndex].SourceLine = forSourceLine;
			loopBegin = new LoopBeginInstruction(registry.PreviousRegister, customVariableNames);
			loopBegin.SourceLine = forSourceLine;
			instructions.Add(loopBegin);
		}
		var bodyAggregatedDirectly =
			GenerateInstructionsForLoopBody(forExpression, aggregationTarget, aggregation);
		if (!string.IsNullOrWhiteSpace(aggregationTarget) && !bodyAggregatedDirectly)
			AddLoopAggregation(aggregationTarget, aggregation);
		var loopEnd = new LoopEndInstruction(instructions.Count - instructionCountBeforeLoopStart)
		{
			Begin = loopBegin,
			SourceLine = forSourceLine
		};
		instructions.Add(loopEnd);
	}

	private void AddLoopAggregation(string aggregationTarget, LoopAggregation aggregation)
	{
		switch (aggregation)
		{
		case LoopAggregation.Number:
			AddNumberAggregation(aggregationTarget);
			break;
		case LoopAggregation.List:
			AddListAggregation(aggregationTarget);
			break;
		}
	}

	private void AddNumberAggregation(string aggregationTarget)
	{
		var loopValueRegister = registry.PreviousRegister;
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), aggregationTarget));
		var accumulatorRegister = registry.PreviousRegister;
		instructions.Add(new BinaryInstruction(InstructionType.Add, accumulatorRegister,
			loopValueRegister, registry.AllocateRegister()));
		instructions.Add(new StoreFromRegisterInstruction(registry.PreviousRegister, aggregationTarget));
	}

	private void AddListAggregation(string aggregationTarget) =>
		instructions.Add(new WriteToListInstruction(registry.PreviousRegister, aggregationTarget));

	private static Expression GetLoopIteratorExpression(Expression iterator)
	{
		if (iterator.ReturnType.IsList || iterator.ReturnType.IsText || iterator.ReturnType.IsNumber ||
			iterator is MethodCall { ReturnType.Name: Type.Range, Method.Name: Method.From })
			return iterator;
		var iteratorMethod = iterator.ReturnType.Methods.FirstOrDefault(method =>
			method.Name == Keyword.For && method.ReturnType.IsIterator);
		return iteratorMethod == null
			? iterator
			: new MethodCall(iteratorMethod, iterator, lineNumber: iterator.LineNumber);
	}

	private LoopBeginInstruction GenerateInstructionForRangeLoopInstruction(
		MethodCall rangeExpression, params string[] customVariableNames)
	{
		GenerateInstructionFromExpression(rangeExpression.Arguments[0]);
		var startIndexRegister = registry.PreviousRegister;
		GenerateInstructionFromExpression(rangeExpression.Arguments[1]);
		var endIndexRegister = registry.PreviousRegister;
		var loopBegin = new LoopBeginInstruction(startIndexRegister, endIndexRegister,
			customVariableNames);
		instructions.Add(loopBegin);
		return loopBegin;
	}

	private bool GenerateInstructionsForLoopBody(For forExpression, string? aggregationTarget,
		LoopAggregation aggregation)
	{
		if (aggregation == LoopAggregation.List && !string.IsNullOrWhiteSpace(aggregationTarget) &&
			forExpression.Body is For directNestedFor)
		{
			GenerateLoopInstructions(directNestedFor, aggregationTarget, aggregation);
			return true;
		}
		if (forExpression.Body is Body forExpressionBody)
		{
			for (var expressionIndex = 0; expressionIndex < forExpressionBody.Expressions.Count;
				expressionIndex++)
			{
				var expression = forExpressionBody.Expressions[expressionIndex];
				if (aggregation == LoopAggregation.List &&
					expressionIndex == forExpressionBody.Expressions.Count - 1 &&
					expression is For nestedFor && !string.IsNullOrWhiteSpace(aggregationTarget))
				{
					GenerateLoopInstructions(nestedFor, aggregationTarget, aggregation);
					return true;
				}
				GenerateInstructionFromExpression(expression);
			}
		}
		else
			GenerateInstructionFromExpression(forExpression.Body);
		return false;
	}

	private void GenerateIfInstructions(If ifExpression)
	{
		GenerateCodeForIfCondition(ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
		if (ifExpression.OptionalElse == null)
			return;
		idStack.Push(conditionalId);
		instructions.Add(new JumpToId(conditionalId++, InstructionType.JumpToIdIfTrue));
		GenerateInstructions([ifExpression.OptionalElse]);
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
	}

	private void GenerateCodeForThen(If ifExpression)
	{
		if (ifExpression.Then is Body thenBody)
			GenerateInstructions(thenBody.Expressions);
		else
			GenerateInstructions([ifExpression.Then]);
	}

	private void GenerateCodeForBinary(MethodCall binaryExpression)
	{
		if (CanGenerateDirectBinaryInstruction(binaryExpression.Method.Name))
			GenerateBinaryInstruction(binaryExpression,
				GetInstructionBasedOnBinaryOperationName(binaryExpression.Method.Name));
	}

	private static bool CanGenerateDirectBinaryInstruction(string methodName) =>
		methodName is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
			or BinaryOperator.Divide or BinaryOperator.Modulate;

	private static InstructionType GetInstructionBasedOnBinaryOperationName(string binaryOperator) =>
		binaryOperator switch
		{
			BinaryOperator.Plus => InstructionType.Add,
			BinaryOperator.Multiply => InstructionType.Multiply,
			BinaryOperator.Minus => InstructionType.Subtract,
			BinaryOperator.Divide => InstructionType.Divide,
			BinaryOperator.Modulate => InstructionType.Modulo,
			_ => throw new NotImplementedException() //ncrunch: no coverage
		};

	private void GenerateCodeForIfCondition(Expression condition)
	{
		if (condition is MethodCall binaryCondition && IsBinaryComparison(binaryCondition))
			GenerateForBinaryIfConditionalExpression(binaryCondition);
		else
			GenerateForBooleanCallIfCondition(condition);
	}

	private static bool IsBinaryComparison(MethodCall call) =>
		call.Method.Name is BinaryOperator.Is or BinaryOperator.Greater
			or BinaryOperator.GreaterOrEqual or BinaryOperator.Smaller or BinaryOperator.SmallerOrEqual
			or BinaryOperator.In || call.Method.Name.StartsWith("is not", StringComparison.Ordinal);

	private void GenerateForBinaryIfConditionalExpression(MethodCall condition)
	{
		var leftRegister = GenerateLeftSideForIfCondition(condition);
		var rightRegister = GenerateRightSideForIfCondition(condition);
		GenerateInstructionsFromIfCondition(GetConditionalInstruction(condition.Method), leftRegister,
			rightRegister);
	}

	private Register GenerateLeftSideForIfCondition(MethodCall condition) =>
		condition.Instance switch
		{
			MethodCall nestedMethodCall when IsBinaryOperation(nestedMethodCall.Method.Name) =>
				GenerateValueBinaryInstructions(nestedMethodCall,
					GetInstructionBasedOnBinaryOperationName(nestedMethodCall.Method.Name)),
			MethodCall nestedMethodCall => InvokeAndGetStoredRegisterForConditional(nestedMethodCall),
			_ => LoadVariableForIfConditionLeft(condition)
		};

	private static bool IsBinaryOperation(string methodName) =>
		methodName is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
			or BinaryOperator.Divide or BinaryOperator.Modulate;

	private Register InvokeAndGetStoredRegisterForConditional(MethodCall condition)
	{
		GenerateInstructionFromExpression(condition);
		return registry.PreviousRegister;
	}

	private Register GenerateRightSideForIfCondition(MethodCall condition)
	{
		GenerateInstructionFromExpression(condition.Arguments[0]);
		return registry.PreviousRegister;
	}

	private void GenerateBinaryInstruction(MethodCall binaryExpression,
		InstructionType operationInstruction)
	{
		if (binaryExpression.Instance is MethodCall nestedBinary &&
			CanGenerateDirectBinaryInstruction(nestedBinary.Method.Name))
		{
			var leftRegister = GenerateValueBinaryInstructions(nestedBinary,
				GetInstructionBasedOnBinaryOperationName(nestedBinary.Method.Name));
			GenerateInstructionFromExpression(binaryExpression.Arguments[0]);
			instructions.Add(new BinaryInstruction(operationInstruction, leftRegister,
				registry.PreviousRegister, registry.AllocateRegister()));
		}
		else if (binaryExpression.Arguments[0] is MethodCall nestedBinaryArgument &&
			CanGenerateDirectBinaryInstruction(nestedBinaryArgument.Method.Name))
			GenerateNestedBinaryInstructions(binaryExpression, operationInstruction,
				nestedBinaryArgument);
		else
			GenerateValueBinaryInstructions(binaryExpression, operationInstruction);
	}

	private void GenerateNestedBinaryInstructions(MethodCall binaryExpression,
		InstructionType operationInstruction, MethodCall binaryArgument)
	{
		var right = GenerateValueBinaryInstructions(binaryArgument,
			GetInstructionBasedOnBinaryOperationName(binaryArgument.Method.Name));
		var left = registry.AllocateRegister();
		if (binaryExpression.Instance != null)
			instructions.Add(new LoadVariableToRegister(left, binaryExpression.Instance.ToString()));
		instructions.Add(new BinaryInstruction(operationInstruction, left, right,
			registry.AllocateRegister()));
	}

	private Register GenerateValueBinaryInstructions(MethodCall binaryExpression,
		InstructionType operationInstruction)
	{
		if (binaryExpression.Instance == null)
			throw new InstanceNameNotFound();
		GenerateInstructionFromExpression(binaryExpression.Instance);
		var leftValue = registry.PreviousRegister;
		GenerateInstructionFromExpression(binaryExpression.Arguments[0]);
		var rightValue = registry.PreviousRegister;
		var resultRegister = registry.AllocateRegister();
		instructions.Add(new BinaryInstruction(operationInstruction, leftValue, rightValue,
			resultRegister));
		return resultRegister;
	}

	private sealed class InstanceNameNotFound : Exception;

	private Dictionary<string, Dictionary<string, List<BinaryMethod>>> GenerateRunMethods(
		IReadOnlyList<Method> runMethods, Type entryType)
	{
		var methodsByType =
			new Dictionary<string, Dictionary<string, List<BinaryMethod>>>(StringComparer.Ordinal);
		var methodsToCompile = new Queue<Method>();
		var compiledMethodKeys = new HashSet<string>(StringComparer.Ordinal);
		foreach (var runMethod in runMethods)
		{
			CollectMethodDependencies(runMethod);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
			var methodBody = runMethod.GetBodyAndParseIfNeeded();
			var methodExpressions = methodBody is Body body
				? body.Expressions
				: [methodBody];
			var childGenerator = new BinaryGenerator(binary.basePackage, methodExpressions,
				runMethod.ReturnType);
			var methodInstructions = childGenerator.GenerateInstructionList();
			var parameters = CreateBinaryMembers(runMethod.Parameters, entryType);
			AddCompiledMethod(methodsByType, runMethod.Type.FullName, runMethod.Name, parameters,
				GetBinaryTypeName(runMethod.ReturnType, entryType), methodInstructions);
			compiledMethodKeys.Add(BuildMethodKey(runMethod));
			EnqueueDiscoveredMethods(childGenerator.DiscoveredInvokeMethods, methodsToCompile,
				compiledMethodKeys);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
		}
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			CollectMethodDependencies(method);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
			var body = method.GetBodyAndParseIfNeeded();
			var methodExpressions = body is Body methodBody
				? methodBody.Expressions
				: [body];
				var childGenerator = new BinaryGenerator(binary.basePackage, methodExpressions,
				method.ReturnType);
			var methodInstructions = childGenerator.GenerateInstructionList();
			var parameters = CreateBinaryMembers(method.Parameters, entryType);
			AddCompiledMethod(methodsByType, method.Type.FullName, method.Name, parameters,
				GetBinaryTypeName(method.ReturnType, entryType), methodInstructions);
			EnqueueDiscoveredMethods(childGenerator.DiscoveredInvokeMethods, methodsToCompile,
				compiledMethodKeys);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
		}
		return methodsByType;
	}

	private static List<BinaryMember> CreateBinaryMembers(IReadOnlyList<Parameter> parameters,
		Type entryType) =>
		parameters.Select(parameter =>
				new BinaryMember(parameter.Name, GetBinaryTypeName(parameter.Type, entryType), null)).
			ToList();

	private void AddGeneratedTypes(
		Dictionary<string, Dictionary<string, List<BinaryMethod>>> methodsByType, Type entryType)
	{
		var orderedTypes = dependencyTypes.Values.OrderBy(type =>
			type == entryType
				? string.Empty
				: GetBinaryTypeName(type, entryType), StringComparer.Ordinal);
		foreach (var type in orderedTypes)
		{
			var members = type.Members.
				Where(member => !member.IsConstant || member.InitialValue != null).Select(member =>
					new BinaryMember(member.Name, GetBinaryTypeName(member.Type, entryType),
						CreateInitialValueInstruction(member.InitialValue))).ToList();
			binary.AddType(GetBinaryTypeName(type, entryType), members,
				methodsByType.TryGetValue(type.FullName, out var methodGroups)
					? methodGroups
					: new Dictionary<string, List<BinaryMethod>>(StringComparer.Ordinal),
				type == entryType);
		}
	}

	private static Instruction? CreateInitialValueInstruction(Expression? initialValue) =>
		initialValue switch
		{
			List list when list.TryGetConstantData() is { } listValue => new SetInstruction(listValue,
				Register.R0),
			Value value => new SetInstruction(value.Data, Register.R0),
			_ => null
		};

	private void CollectMethodDependencies(Method method)
	{
		CollectTypeDependency(method.Type, true);
		CollectTypeDependency(method.ReturnType, false);
		foreach (var parameter in method.Parameters)
			CollectTypeDependency(parameter.Type, false);
		if (method.Type.IsTrait)
			return;
		var body = method.GetBodyAndParseIfNeeded();
		if (body is Body methodBody)
			foreach (var expression in methodBody.Expressions)
				CollectExpressionDependencies(expression);
		else
			CollectExpressionDependencies(body);
	}

	private void CollectExpressionDependencies(Expression expression)
	{
		CollectTypeDependency(expression.ReturnType, false);
		switch (expression)
		{
		case Body body:
			foreach (var child in body.Expressions)
				CollectExpressionDependencies(child);
			break;
		case Binary binaryExpr:
			CollectTypeDependency(binaryExpr.Method.Type, true);
			CollectTypeDependency(binaryExpr.Method.ReturnType, false);
			foreach (var parameter in binaryExpr.Method.Parameters)
				CollectTypeDependency(parameter.Type, false);
			CollectExpressionDependencies(binaryExpr.Instance!);
			// ReSharper disable TailRecursiveCall
			CollectExpressionDependencies(binaryExpr.Arguments[0]);
			break;
		case Declaration declaration:
			CollectExpressionDependencies(declaration.Value);
			break;
		case MutableReassignment reassignment:
			CollectExpressionDependencies(reassignment.Value);
			break;
		case For forExpression:
			CollectExpressionDependencies(GetLoopIteratorExpression(forExpression.Iterator));
			CollectExpressionDependencies(forExpression.Body);
			break;
		case If ifExpression:
			CollectExpressionDependencies(ifExpression.Condition);
			CollectExpressionDependencies(ifExpression.Then);
			if (ifExpression.OptionalElse != null)
				CollectExpressionDependencies(ifExpression.OptionalElse);
			break;
		case SelectorIf selectorIf:
			CollectExpressionDependencies(selectorIf.Selector);
			foreach (var @case in selectorIf.Cases)
			{
				CollectExpressionDependencies(@case.Pattern);
				CollectExpressionDependencies(@case.Then);
			}
			if (selectorIf.OptionalElse != null)
				CollectExpressionDependencies(selectorIf.OptionalElse);
			break;
		case ListCall listCall:
			CollectExpressionDependencies(listCall.List);
			CollectExpressionDependencies(listCall.Index);
			break;
		case MemberCall memberCall:
			CollectTypeDependency(memberCall.Member.Type, false);
			if (memberCall.Instance != null)
				CollectExpressionDependencies(memberCall.Instance);
			break;
		case MethodCall methodCall:
			CollectTypeDependency(methodCall.Method.Type, true);
			CollectTypeDependency(methodCall.Method.ReturnType, false);
			foreach (var parameter in methodCall.Method.Parameters)
				CollectTypeDependency(parameter.Type, false);
			if (methodCall.Instance != null)
				CollectExpressionDependencies(methodCall.Instance);
			foreach (var argument in methodCall.Arguments)
				CollectExpressionDependencies(argument);
			break;
		}
	}

	private void CollectTypeDependency(Type type, bool includeType)
	{
		if (type.IsNone || type.IsAny)
			return;
		if (type is GenericTypeImplementation genericImplementation)
		{
			foreach (var implementationType in genericImplementation.ImplementationTypes)
				CollectTypeDependency(implementationType, false);
			if (!includeType)
				return;
		}
		else if (type is GenericType genericType)
		{
			foreach (var implementation in genericType.GenericImplementations)
				CollectTypeDependency(implementation.Type, false);
			if (!includeType)
				return;
		}
		if (!dependencyTypes.TryAdd(type.FullName, type))
			return;
		foreach (var member in type.Members)
			CollectTypeDependency(member.Type, false);
	}

	private static string GetBinaryTypeName(Type type, Type entryType)
	{
		if (IsStrictBaseType(type, entryType))
			return nameof(Strict) + Context.ParentSeparator + type.Name;
		var entryPackagePrefix = entryType.Package.FullName + Context.ParentSeparator;
		return type.FullName.StartsWith(entryPackagePrefix, StringComparison.Ordinal)
			? type.FullName[entryPackagePrefix.Length..]
			: type.FullName;
	}

	private static bool IsStrictBaseType(Type type, Type entryType) =>
		type.FullName != entryType.FullName && (type.Package.Name == nameof(Strict) ||
			entryType.Package.Name == "TestPackage" && type.Package.Name == "TestPackage");

	private Dictionary<string, Dictionary<string, List<BinaryMethod>>> CompileMethodsFromExpressions(
		string thisEntryTypeFullName, IReadOnlyList<Expression> entryExpressions, Type runReturnType)
	{
		var methodsByType = new Dictionary<string, Dictionary<string, List<BinaryMethod>>>(
			StringComparer.Ordinal);
		var methodsToCompile = new Queue<Method>();
		var compiledMethodKeys = new HashSet<string>(StringComparer.Ordinal);
		var runInstructions = GenerateInstructions(entryExpressions);
		AddCompiledMethod(methodsByType, thisEntryTypeFullName, Method.Run, [], runReturnType.Name,
			runInstructions);
		EnqueueDiscoveredMethods(discoveredInvokeMethods, methodsToCompile, compiledMethodKeys);
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			CollectMethodDependencies(method);
			var body = method.GetBodyAndParseIfNeeded();
			var methodExpressions = body is Body methodBody
				? methodBody.Expressions
				: [body];
			var childGenerator = new BinaryGenerator(binary.basePackage, methodExpressions,
				method.ReturnType);
			var methodInstructions = childGenerator.GenerateInstructionList();
			var parameters = method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToList();
			AddCompiledMethod(methodsByType, method.Type.FullName, method.Name, parameters,
				method.ReturnType.Name, methodInstructions);
			EnqueueDiscoveredMethods(childGenerator.DiscoveredInvokeMethods, methodsToCompile,
				compiledMethodKeys);
		}
		return methodsByType;
	}

	//TODO: remove
	private List<Instruction> GenerateInstructionList() => GenerateInstructions(Expressions);

	private void EnqueueConstraintMethods(Queue<Method> methodsToCompile,
		HashSet<string> compiledMethodKeys)
	{
		foreach (var type in dependencyTypes.Values)
			foreach (var member in type.Members)
				EnqueueConstraintMethods(type, member, methodsToCompile, compiledMethodKeys);
	}

	private static void EnqueueConstraintMethods(Type type, Member member,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		if (member.Constraints == null)
			return;
		foreach (var constraint in member.Constraints)
			if (constraint is Binary { Method.Name: BinaryOperator.Is, Instance: { } instance } binary &&
				instance.ToString() == "Length")
			{
				var rhsText = binary.Arguments[0].ToString();
				var separatorIndex = rhsText.IndexOf('.');
				if (separatorIndex <= 0)
					continue;
				var referencedMember = type.Members.FirstOrDefault(typeMember =>
					typeMember.Name.Equals(rhsText[..separatorIndex], StringComparison.OrdinalIgnoreCase));
				var method = referencedMember?.Type.FindMethod(rhsText[(separatorIndex + 1)..], []);
				if (method != null && compiledMethodKeys.Add(BuildMethodKey(method)))
					methodsToCompile.Enqueue(method);
			}
	}

	private static string BuildMethodKey(Method method) =>
		method.Type.FullName + ":" + BinaryExecutable.BuildMethodHeader(method.Name,
			method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToArray(),
			method.ReturnType);

	private static void EnqueueDiscoveredMethods(IReadOnlyList<Method> methods,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		foreach (var method in methods)
		{
			var methodKey = BuildMethodKey(method);
			if (compiledMethodKeys.Add(methodKey))
				methodsToCompile.Enqueue(method);
		}
	}

	private static void AddCompiledMethod(
		Dictionary<string, Dictionary<string, List<BinaryMethod>>> methodsByType,
		string typeFullName, string methodName, List<BinaryMember> parameters,
		string returnTypeName, List<Instruction> instructionsToAdd)
	{
		if (!methodsByType.TryGetValue(typeFullName, out var methodGroups))
		{
			methodGroups = new Dictionary<string, List<BinaryMethod>>(StringComparer.Ordinal);
			methodsByType[typeFullName] = methodGroups;
		}
		if (!methodGroups.TryGetValue(methodName, out var overloads))
		{
			overloads = [];
			methodGroups[methodName] = overloads;
		}
		overloads.Add(new BinaryMethod(methodName, parameters, returnTypeName, instructionsToAdd));
	}

	private static string ExtractTextPrefix(Expression? expression) =>
		expression switch
		{
			Value value when value.Data.IsText => value.Data.Text,
			To { Instance: { } inner } => ExtractTextPrefix(inner),
			_ => ""
		};

	private static Expression UnwrapToConversion(Expression expression) =>
		expression is To { Instance: { } inner }
			? inner
			: expression;

	private bool TryGenerateInstructionForCollectionManipulation(MethodCall methodCall)
	{
		switch (methodCall.Method.Name)
		{
		case "Add" when methodCall.Instance?.ReturnType.IsList == true ||
			methodCall.Instance?.ReturnType.IsDictionary == true:
			GenerateInstructionsForAddMethod(methodCall);
			return true;
		case "Remove" when methodCall.Instance?.ReturnType.IsList == true:
			GenerateInstructionsForRemoveMethod(methodCall);
			return true;
		case "Increment":
		case "Decrement":
			GenerateIncrementDecrementInvoke(methodCall);
			return true;
		default:
			return false;
		}
	}

	private void GenerateIncrementDecrementInvoke(MethodCall methodCall)
	{
		Register? instanceRegister = null;
		if (methodCall.Instance != null)
		{
			GenerateInstructionFromExpression(methodCall.Instance);
			instanceRegister = registry.PreviousRegister;
		}
		var parameterNames = new string[methodCall.Method.Parameters.Count];
		for (var paramIndex = 0; paramIndex < methodCall.Method.Parameters.Count; paramIndex++)
			parameterNames[paramIndex] = methodCall.Method.Parameters[paramIndex].Name;
		var methodInfo = new InvokeMethodInfo(
			methodCall.Method.Type.FullName,
			methodCall.Method.Name, parameterNames,
			GetBinaryTypeName(methodCall.ReturnType, methodCall.Method.Type),
			[], instanceRegister);
		var resultRegister = registry.AllocateRegister();
		instructions.Add(new Invoke(resultRegister, methodInfo));
		if (methodCall.Instance != null)
			instructions.Add(new StoreFromRegisterInstruction(resultRegister,
				methodCall.Instance.ToString()));
	}

	private void GenerateInstructionsForRemoveMethod(MethodCall methodCall)
	{
		if (methodCall.Instance == null)
			return;
		GenerateInstructionFromExpression(methodCall.Arguments[0]);
		if (methodCall.Instance.ReturnType.IsList)
			instructions.Add(new RemoveInstruction(registry.PreviousRegister, methodCall.Instance.ToString()));
	}

	private void GenerateInstructionsForAddMethod(MethodCall methodCall)
	{
		if (TryGenerateAddForTable(methodCall) || methodCall.Instance == null)
			return;
		GenerateInstructionFromExpression(methodCall.Arguments[0]);
		instructions.Add(new WriteToListInstruction(registry.PreviousRegister,
			methodCall.Instance.ToString()));
	}

	private bool TryGenerateAddForTable(MethodCall methodCall)
	{
		if (methodCall.Arguments.Count != 2 || methodCall.Instance == null)
			return false;
		GenerateInstructionFromExpression(methodCall.Arguments[0]);
		var key = registry.PreviousRegister;
		GenerateInstructionFromExpression(methodCall.Arguments[1]);
		var value = registry.PreviousRegister;
		instructions.Add(new WriteToTableInstruction(key, value, methodCall.Instance.ToString()));
		return true;
	}

	private void GenerateForAssignmentOrDeclaration(Expression declarationOrAssignment,
		string name)
	{
		if (declarationOrAssignment is Value declarationOrAssignmentValue)
			TryGenerateInstructionsForAssignmentValue(declarationOrAssignmentValue, name);
		else
		{
			GenerateInstructionFromExpression(declarationOrAssignment);
			instructions.Add(new StoreFromRegisterInstruction(registers[registry.NextRegister - 1],
				name));
		}
	}

	private void TryGenerateInstructionsForAssignmentValue(Value assignmentValue,
		string variableName)
	{
		var data = assignmentValue.ReturnType.IsDictionary
			? new ValueInstance(assignmentValue.ReturnType,
				new Dictionary<ValueInstance, ValueInstance>())
			: GetValueInstanceFromExpression(assignmentValue);
		instructions.Add(new StoreVariableInstruction(data, variableName));
	}

	private void GenerateSelectorIfInstructions(SelectorIf selectorIf)
	{
		foreach (var selectorCase in selectorIf.Cases)
		{
			GenerateCodeForIfCondition(selectorCase.Condition);
			GenerateInstructionFromExpression(selectorCase.Then);
			instructions.Add(new ReturnInstruction(registry.PreviousRegister));
			instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
		}
		if (selectorIf.OptionalElse != null)
		{
			GenerateInstructionFromExpression(selectorIf.OptionalElse);
			instructions.Add(new ReturnInstruction(registry.PreviousRegister));
		}
	}

	private void GenerateForBooleanCallIfCondition(Expression condition)
	{
		GenerateInstructionFromExpression(condition);
		var instanceCallRegister = registry.PreviousRegister;
		instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
			new ValueInstance(condition.ReturnType, 1.0)));
		GenerateInstructionsFromIfCondition(InstructionType.Equal, instanceCallRegister,
			registry.PreviousRegister);
	}

	private void GenerateInstructionsFromIfCondition(InstructionType conditionInstruction,
		Register leftRegister, Register rightRegister)
	{
		instructions.Add(new BinaryInstruction(conditionInstruction, leftRegister, rightRegister));
		idStack.Push(conditionalId);
		instructions.Add(new JumpToId(conditionalId++, InstructionType.JumpToIdIfFalse));
	}

	private static InstructionType GetConditionalInstruction(Method condition) =>
		condition.Name switch
		{
			BinaryOperator.Greater => InstructionType.GreaterThan,
			BinaryOperator.Smaller => InstructionType.LessThan,
			_ => InstructionType.Equal
		};

	private Register LoadVariableForIfConditionLeft(MethodCall condition)
	{
		if (condition.Instance != null)
			GenerateInstructionFromExpression(condition.Instance);
		return registry.PreviousRegister;
	}
}