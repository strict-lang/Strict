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
	//TODO: not really used,  these are all wrong, the constructor should only use the basePackage to make everything possible, the Generate method should get the entry point and find the rest from there! 3 constructors is just plain stupid. this was mostly to get the old tests working, but they are mostly wrong anyway!
	public BinaryGenerator(Expression entryPoint)
	{
		entryTypeFullName = GetEntryTypeFullName(entryPoint);
		ReturnType = entryPoint.ReturnType;
		Expressions = [entryPoint];
		binary = new BinaryExecutable(GetBasePackage(entryPoint));
	}

	public BinaryGenerator(MethodCall methodCall)
	{
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

	//TODO: way too many fields, this should not all be at class level!
	private readonly BinaryExecutable binary;
	private readonly string entryTypeFullName;
	private readonly List<Instruction> instructions = [];
	private static readonly HashSet<string> StrictRuntimeTypeNames =
	[
		Type.Any,
		Type.Boolean,
		Type.Character,
		Type.Dictionary,
		Type.Logger,
		Type.Number,
		Type.Range,
		Type.Text,
		Type.TextReader,
		Type.TextWriter,
		Type.List,
		Type.None,
		Type.System
	];
	private readonly Dictionary<string, Type> dependencyTypes = new(StringComparer.Ordinal);
	private readonly Registry registry = new();
	private readonly Stack<int> idStack = new();
	private readonly Register[] registers = Enum.GetValues<Register>();
	private IReadOnlyList<Expression> Expressions { get; } //TODO: stupid
	private Type ReturnType { get; } //TODO: forbidden!
	private int conditionalId;
	private int forResultId;

	public BinaryExecutable Generate() =>
		Generate(entryTypeFullName, Expressions, ReturnType);

	public static BinaryExecutable GenerateFromRunMethods(Method preferredEntryMethod,
		IReadOnlyList<Method> runMethods)
	{
		var generator = new BinaryGenerator(GetBasePackage(preferredEntryMethod), [],
			preferredEntryMethod.ReturnType);
		return generator.Generate(preferredEntryMethod, runMethods);
	}

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
		var methodsByType =
			GenerateEntryMethods(typeFullName, entryExpressions, runReturnType);
		foreach (var (compiledTypeFullName, methodGroups) in methodsByType)
			binary.AddType(compiledTypeFullName, [], methodGroups);
		return binary;
	}

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

	private static string GetEntryTypeFullName(Expression expression) =>
		expression is MethodCall methodCall
			? methodCall.Method.Type.FullName
			: expression.ReturnType.FullName;

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
			if (instance.Method.Parameters[parameterIndex].Type is GenericTypeImplementation
				{
					Generic.Name: Type.List
				} && !(instance.Arguments.Count == 1 && instance.Arguments[0] is List))
			{
				//TODO: not very efficient
				var listItems = instance.Arguments.Select(GetValueInstanceFromExpression).ToArray();
				instructions.Add(new StoreVariableInstruction(
					new ValueInstance(instance.Method.Parameters[parameterIndex].Type, listItems),
					instance.ReturnType.Members[parameterIndex].Name, isMember: true));
			}
			else
				instructions.Add(new StoreVariableInstruction(
					GetValueInstanceFromExpression(instance.Arguments[parameterIndex]),
					instance.ReturnType.Members[parameterIndex].Name, isMember: true));
	}

	private void AddMethodParameterVariables(MethodCall methodCall)
	{
		for (var index = 0; index < methodCall.Method.Parameters.Count &&
			index < methodCall.Arguments.Count; index++)
			instructions?.Add(new StoreVariableInstruction(
				GetValueInstanceFromExpression(methodCall.Arguments[index]),
				methodCall.Method.Parameters[index].Name));
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

	private void GenerateInstructionForNumberAggregation(For forExpression)
	{
		var resultVariable = $"forResult{forResultId++}";
		instructions.Add(new StoreVariableInstruction(new ValueInstance(ReturnType, 0), resultVariable));
		GenerateLoopInstructions(forExpression, resultVariable);
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), resultVariable));
		instructions.Add(new ReturnInstruction(registry.PreviousRegister));
	}

	private void GenerateInstructionFromExpression(Expression expression) =>
		_ = TryGenerateBodyInstructions(expression) ?? TryGenerateBinaryInstructions(expression) ??
			TryGenerateIfInstructions(expression) ?? TryGenerateSelectorIfInstructions(expression) ??
			TryGenerateAssignmentInstructions(expression) ??
			TryGenerateLoopInstructions(expression) ?? TryGenerateMutableInstructions(expression) ??
			TryGenerateMemberCallInstruction(expression) ??
			TryGenerateVariableCallInstruction(expression) ?? TryGenerateValueInstruction(expression) ??
			TryGenerateMethodCallInstruction(expression) ?? TryGenerateListCallInstruction(expression) ??
			throw new NotSupportedException(expression.ToString());

	private bool? TryGenerateListCallInstruction(Expression expression)
	{
		if (expression is not ListCall listCall)
			return null; //ncrunch: no coverage
		GenerateInstructionFromExpression(listCall.Index);
		var indexRegister = registry.PreviousRegister;
		instructions.Add(new ListCallInstruction(registry.AllocateRegister(), indexRegister,
			listCall.List.ToString()));
		return true;
	}

	private bool? TryGenerateVariableCallInstruction(Expression expression)
	{
		if (expression is not (VariableCall or ParameterCall or Instance))
			return null;
		instructions.Add(
			new LoadVariableToRegister(registry.AllocateRegister(), expression.ToString()));
		return true;
	}

	private bool? TryGenerateMemberCallInstruction(Expression expression)
	{
		if (expression is not MemberCall memberCall)
			return null;
		if (memberCall.Instance == null)
			instructions.Add(
				new LoadVariableToRegister(registry.AllocateRegister(), expression.ToString()));
		else if (memberCall.Member.InitialValue != null)
			TryGenerateForEnum(memberCall.Instance.ReturnType, memberCall.Member.InitialValue);
		return true;
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

	private bool? TryGenerateValueInstruction(Expression expression)
	{
		if (expression is not Value)
			return null;
		instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
			GetValueInstanceFromExpression(expression)));
		return true;
	}

	private bool? TryGenerateMethodCallInstruction(Expression expression)
	{
		if (expression is not MethodCall methodCall)
			return null;
		if (TryGenerateInstructionForCollectionManipulation(methodCall))
			return true;
		if (TryGeneratePrintInstruction(methodCall))
			return true;
		instructions.Add(new Invoke(registry.AllocateRegister(), methodCall, registry));
		return true;
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

	private bool? TryGenerateBinaryInstructions(Expression expression)
	{
		if (expression is Binary binaryExpression)
		{
			if (binaryExpression.Method.Name == BinaryOperator.Is)
				return true;
			if (!CanGenerateDirectBinaryInstruction(binaryExpression.Method.Name))
				return TryGenerateMethodCallInstruction(binaryExpression);
			GenerateCodeForBinary(binaryExpression);
			return true; //TODO: there is not even false here, this is no good
		}
		return null;
	}

	private void GenerateLoopInstructions(For forExpression, string? aggregationTarget = null)
	{
		var instructionCountBeforeLoopStart = instructions.Count;
		LoopBeginInstruction loopBegin;
		if (forExpression.Iterator is MethodCall rangeExpression &&
			forExpression.Iterator.ReturnType.Name == Type.Range &&
			rangeExpression.Method.Name == Method.From)
			loopBegin = GenerateInstructionForRangeLoopInstruction(rangeExpression);
		else
		{
			GenerateInstructionFromExpression(forExpression.Iterator);
			loopBegin = new LoopBeginInstruction(registry.PreviousRegister);
			instructions.Add(loopBegin);
		}
		GenerateInstructionsForLoopBody(forExpression);
		if (!string.IsNullOrWhiteSpace(aggregationTarget))
			AddNumberAggregation(aggregationTarget);
		instructions.Add(new LoopEndInstruction(instructions.Count - instructionCountBeforeLoopStart)
		{
			Begin = loopBegin
		});
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

	private LoopBeginInstruction GenerateInstructionForRangeLoopInstruction(MethodCall rangeExpression)
	{
		GenerateInstructionFromExpression(rangeExpression.Arguments[0]);
		var startIndexRegister = registry.PreviousRegister;
		GenerateInstructionFromExpression(rangeExpression.Arguments[1]);
		var endIndexRegister = registry.PreviousRegister;
		var loopBegin = new LoopBeginInstruction(startIndexRegister, endIndexRegister);
		instructions.Add(loopBegin);
		return loopBegin;
	}

	private void GenerateInstructionsForLoopBody(For forExpression)
	{
		if (forExpression.Body is Body forExpressionBody)
			GenerateInstructions(forExpressionBody.Expressions);
		else
			GenerateInstructionFromExpression(forExpression.Body);
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
		if (condition is MethodCall binaryCondition)
			GenerateForBinaryIfConditionalExpression(binaryCondition);
		else
			GenerateForBooleanCallIfCondition(condition);
	}

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

		void EnqueueInvokedMethods(IReadOnlyList<Instruction> methodInstructions)
		{
			foreach (var invoke in methodInstructions.OfType<Invoke>())
			{
				if (invoke.Method.Method.Name == Method.From)
					continue;
				var invokedMethod = invoke.Method.Method;
				var methodKey = BuildMethodKey(invokedMethod);
				if (compiledMethodKeys.Add(methodKey))
					methodsToCompile.Enqueue(invokedMethod);
			}
		}

		foreach (var runMethod in runMethods)
		{
			CollectMethodDependencies(runMethod);
			var methodBody = runMethod.GetBodyAndParseIfNeeded();
			var methodExpressions = methodBody is Body body
				? body.Expressions
				: [methodBody];
			var methodInstructions = new BinaryGenerator(binary.basePackage, methodExpressions,
				runMethod.ReturnType).GenerateInstructionList();
			var parameters = CreateBinaryMembers(runMethod.Parameters, entryType);
			AddCompiledMethod(methodsByType, runMethod.Type.FullName, runMethod.Name, parameters,
				GetBinaryTypeName(runMethod.ReturnType, entryType), methodInstructions);
			compiledMethodKeys.Add(BuildMethodKey(runMethod));
			EnqueueInvokedMethods(methodInstructions);
		}
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			CollectMethodDependencies(method);
			var body = method.GetBodyAndParseIfNeeded();
			var methodExpressions = body is Body methodBody
				? methodBody.Expressions
				: [body];
			var methodInstructions = new BinaryGenerator(binary.basePackage, methodExpressions,
				method.ReturnType).GenerateInstructionList();
			var parameters = CreateBinaryMembers(method.Parameters, entryType);
			AddCompiledMethod(methodsByType, method.Type.FullName, method.Name, parameters,
				GetBinaryTypeName(method.ReturnType, entryType), methodInstructions);
			EnqueueInvokedMethods(methodInstructions);
		}
		return methodsByType;
	}

	//TODO: slow, should be optimized! also a binary always has the same structure, why so complicated here?
	private static List<BinaryMember> CreateBinaryMembers(IReadOnlyList<Parameter> parameters,
		Type entryType) =>
		parameters.Select(parameter =>
				new BinaryMember(parameter.Name, GetBinaryTypeName(parameter.Type, entryType), null)).
			ToList();

	private void AddGeneratedTypes(
		Dictionary<string, Dictionary<string, List<BinaryMethod>>> methodsByType,
		Type entryType)
	{
		var orderedTypes = dependencyTypes.Values.OrderBy(type =>
			type == entryType
				? string.Empty
				: GetBinaryTypeName(type, entryType), StringComparer.Ordinal);
		foreach (var type in orderedTypes)
		{
			var members = type.Members
				.Where(member => !member.IsConstant || member.InitialValue != null)
				.Select(member => new BinaryMember(member.Name, GetBinaryTypeName(member.Type, entryType),
					member.IsConstant ? CreateConstantInstruction(member) : null))
				.ToList();
			binary.AddType(GetBinaryTypeName(type, entryType), members,
				methodsByType.TryGetValue(type.FullName, out var methodGroups)
					? methodGroups
					: new Dictionary<string, List<BinaryMethod>>(StringComparer.Ordinal),
				type == entryType);
		}
	}

	private static SetInstruction CreateConstantInstruction(Member member) =>
		member.InitialValue is Value val
			? new SetInstruction(val.Data, Register.R0)
			: new SetInstruction(new ValueInstance(member.InitialValue!.ToString()), Register.R0);

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
			CollectExpressionDependencies(forExpression.Iterator);
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

	private static bool IsStrictBaseType(Type type, Type entryType)
	{
		if (type.FullName == entryType.FullName)
			return false;
		if (type.Package.Name == nameof(Strict))
			return true;
		return entryType.Package.Name == "TestPackage" && type.Package.Name == "TestPackage" &&
			StrictRuntimeTypeNames.Contains(type.Name);
	}

	//TODO: remove, bad naming
	private Dictionary<string, Dictionary<string, List<BinaryMethod>>> GenerateEntryMethods(
		string thisEntryTypeFullName, IReadOnlyList<Expression> entryExpressions, Type runReturnType)
	{
		var methodsByType = new Dictionary<string, Dictionary<string, List<BinaryMethod>>>(
			StringComparer.Ordinal);
		var methodsToCompile = new Queue<Method>();
		var compiledMethodKeys = new HashSet<string>(StringComparer.Ordinal);

		void EnqueueInvokedMethods(IReadOnlyList<Instruction> thisInstructions) //TODO: remove
		{
			foreach (var invoke in thisInstructions.OfType<Invoke>())
			{
				if (invoke.Method.Method.Name == Method.From)
					continue;
				var method = invoke.Method.Method;
				var methodKey = method.Type.FullName + ":" + BinaryExecutable.BuildMethodHeader(method.Name,
					method.Parameters.Select(parameter =>
						new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToList(),
					method.ReturnType);
				if (compiledMethodKeys.Add(methodKey))
					methodsToCompile.Enqueue(method);
			}
		}

		var runInstructions = GenerateInstructions(entryExpressions);
		AddCompiledMethod(methodsByType, thisEntryTypeFullName, Method.Run, [], runReturnType.Name,
			runInstructions);
		EnqueueInvokedMethods(runInstructions);
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			var body = method.GetBodyAndParseIfNeeded();
			var methodExpressions = body is Body methodBody
				? methodBody.Expressions
				: [body];
			var methodInstructions = new BinaryGenerator(binary.basePackage, methodExpressions,
				method.ReturnType).GenerateInstructionList();
			var parameters = method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToList();
			AddCompiledMethod(methodsByType, method.Type.FullName, method.Name, parameters,
				method.ReturnType.Name, methodInstructions);
			EnqueueInvokedMethods(methodInstructions);
		}
		return methodsByType;
	}

	private List<Instruction> GenerateInstructionList() => GenerateInstructions(Expressions);

	private static string BuildMethodKey(Method method) =>
		method.Type.FullName + ":" + BinaryExecutable.BuildMethodHeader(method.Name,
			method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToList(),
			method.ReturnType);

	private static void EnqueueCalledMethods(IReadOnlyList<Instruction> instructions,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		foreach (var invoke in instructions.OfType<Invoke>())
		{
			if (invoke.Method.Method.Name == Method.From)
				continue;
			var method = invoke.Method.Method;
			var methodKey = method.Type.FullName + ":" + BinaryExecutable.BuildMethodHeader(method.Name,
				method.Parameters.Select(parameter =>
					new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToList(),
				method.ReturnType);
			if (compiledMethodKeys.Add(methodKey))
				methodsToCompile.Enqueue(method);
		}
	}

	//TODO: cumbersome, remove and fix
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
			var register = registry.AllocateRegister();
			instructions.Add(new Invoke(register, methodCall, registry));
			if (methodCall.Instance != null)
				instructions.Add(new StoreFromRegisterInstruction(register, methodCall.Instance.ToString()));
			return true;
		default:
			return false;
		}
	}

	private void GenerateInstructionsForRemoveMethod(MethodCall methodCall)
	{
		if (methodCall.Instance == null)
			return;
		GenerateInstructionFromExpression(methodCall.Arguments[0]);
		if (methodCall.Instance.ReturnType is GenericTypeImplementation { Generic.Name: Type.List })
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

	private bool? TryGenerateBodyInstructions(Expression expression)
	{
		if (expression is not Body body)
			return null;
		GenerateInstructions(body.Expressions);
		return true;
	}

	private bool? TryGenerateMutableInstructions(Expression expression)
	{
		if (expression is Declaration { IsMutable: true } declaration)
			GenerateForAssignmentOrDeclaration(declaration.Value, declaration.Name);
		else if (expression is MutableReassignment assignment)
			GenerateForAssignmentOrDeclaration(assignment.Value, assignment.Name);
		else
			return null;
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

	private bool? TryGenerateLoopInstructions(Expression expression)
	{
		if (expression is not For forExpression)
			return null;
		GenerateLoopInstructions(forExpression);
		return true;
	}

	private bool? TryGenerateAssignmentInstructions(Expression expression)
	{
		if (expression is not Declaration assignmentExpression || expression.IsMutable)
			return null;
		GenerateForAssignmentOrDeclaration(assignmentExpression.Value, assignmentExpression.Name);
		return true;
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

	private bool? TryGenerateIfInstructions(Expression expression)
	{
		if (expression is not If ifExpression)
			return null;
		GenerateIfInstructions(ifExpression);
		return true;
	}

	private bool? TryGenerateSelectorIfInstructions(Expression expression)
	{
		if (expression is not SelectorIf selectorIf)
			return null;
		GenerateSelectorIfInstructions(selectorIf);
		return true;
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