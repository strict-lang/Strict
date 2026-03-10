using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Binary = Strict.Expressions.Binary;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

public sealed class BytecodeGenerator
{
	public BytecodeGenerator(InvokedMethod method, Registry registry)
	{
		foreach (var argument in method.Arguments)
			instructions.Add(new StoreVariableInstruction(argument.Value, argument.Key));
		Expressions = method.Expressions;
		this.registry = registry;
		ReturnType = method.ReturnType;
		if (method is InstanceInvokedMethod instanceMethod)
			AddMembersFromCaller(instanceMethod.InstanceCall);
	}

	private readonly List<Instruction> instructions = [];
	private readonly Registry registry;
	private readonly Stack<int> idStack = new();
	private readonly Register[] registers = Enum.GetValues<Register>();
	private int conditionalId;

	public BytecodeGenerator(MethodCall methodCall)
	{
		if (methodCall.Instance != null)
			AddInstanceMemberVariables((MethodCall)methodCall.Instance);
		AddMethodParameterVariables(methodCall);
		var methodBody = methodCall.Method.GetBodyAndParseIfNeeded();
		Expressions = methodBody is Body body
			? body.Expressions
			: [methodBody];
		registry = new Registry();
		ReturnType = methodCall.Method.ReturnType;
	}

	private IReadOnlyList<Expression> Expressions { get; }
	private Type ReturnType { get; }
	private int forResultId;

	private void AddMembersFromCaller(ValueInstance instance)
	{
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null)
		{
			var members = typeInstance.ReturnType.Members;
			for (var i = 0; i < members.Count && i < typeInstance.Values.Length; i++)
				if (!members[i].Type.IsTrait)
					instructions.Add(new StoreVariableInstruction(typeInstance.Values[i],
						members[i].Name, isMember: true));
			return;
		}
		var firstNonTraitMember = instance.GetTypeExceptText().Members
			.FirstOrDefault(member => !member.Type.IsTrait);
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
			{
				instructions.Add(new StoreVariableInstruction(
					GetValueInstanceFromExpression(instance.Arguments[parameterIndex]),
					instance.ReturnType.Members[parameterIndex].Name, isMember: true));
			}
	}

	private void AddMethodParameterVariables(MethodCall methodCall)
	{
		for (var parameterIndex = 0; parameterIndex < methodCall.Method.Parameters.Count;
			parameterIndex++)
			instructions?.Add(new StoreVariableInstruction(
				GetValueInstanceFromExpression(methodCall.Arguments[parameterIndex]),
				methodCall.Method.Parameters[parameterIndex].Name));
	}

	public List<Instruction> Generate() => GenerateInstructions(Expressions);

	private List<Instruction> GenerateInstructions(IReadOnlyList<Expression> expressions)
	{
		for (var i = 0; i < expressions.Count; i++)
			if ((ReferenceEquals(expressions[i], Expressions[^1]) ||
					expressions[i] is Expressions.Return) && expressions[i] is not If)
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
			TryGenerateIfInstructions(expression) ?? TryGenerateAssignmentInstructions(expression) ??
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
		if (expression is not (VariableCall or ParameterCall))
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
		if (expression is Binary || expression is not MethodCall methodCall)
			return null;
		if (TryGenerateInstructionForCollectionManipulation(methodCall))
			return true;
		instructions.Add(new Invoke(registry.AllocateRegister(), methodCall, registry));
		return true;
	}

	private bool TryGenerateInstructionForCollectionManipulation(MethodCall methodCall)
	{
		switch (methodCall.Method.Name)
		{
		case "Add":
		{
			GenerateInstructionsForAddMethod(methodCall);
			return true;
		}
		case "Remove":
		{
			GenerateInstructionsForRemoveMethod(methodCall);
			return true;
		}
		case "Increment":
		case "Decrement":
		{
			var register = registry.AllocateRegister();
			instructions.Add(new Invoke(register, methodCall, registry));
			if (methodCall.Instance != null)
				instructions.Add(new StoreFromRegisterInstruction(register, methodCall.Instance.ToString()));
			return true;
		}
		default:
			return false;
		}
	}

	private void GenerateInstructionsForRemoveMethod(MethodCall methodCall)
	{
		if (methodCall.Instance == null)
			return; //ncrunch: no coverage
		GenerateInstructionFromExpression(methodCall.Arguments[0]);
		if (methodCall.Instance.ReturnType is GenericTypeImplementation { Generic.Name: Type.List })
			instructions.Add(new RemoveInstruction(methodCall.Instance.ToString(),
				registry.PreviousRegister));
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

	private void GenerateForAssignmentOrDeclaration(Expression declarationOrAssignment, string name)
	{
		if (declarationOrAssignment is Value declarationOrAssignmentValue)
			TryGenerateInstructionsForAssignmentValue(declarationOrAssignmentValue, name);
		else
		{
			GenerateInstructionFromExpression(declarationOrAssignment);
			instructions.Add(new StoreFromRegisterInstruction(registers[registry.NextRegister - 1], name));
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

	private void TryGenerateInstructionsForAssignmentValue(Value assignmentValue, string variableName)
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

	private bool? TryGenerateBinaryInstructions(Expression expression)
	{
		if (expression is not Binary binary)
			return null;
		GenerateCodeForBinary(binary);
		return true;
	}

	private void GenerateLoopInstructions(For forExpression, string? aggregationTarget = null)
	{
		var instructionCountBeforeLoopStart = instructions.Count;
		if (forExpression.Iterator is MethodCall rangeExpression &&
			forExpression.Iterator.ReturnType.Name == Type.Range &&
			rangeExpression.Method.Name == Method.From)
			GenerateInstructionForRangeLoopInstruction(rangeExpression);
		else
		{
			GenerateInstructionFromExpression(forExpression.Iterator);
			instructions.Add(new LoopBeginInstruction(registry.PreviousRegister));
		}
		GenerateInstructionsForLoopBody(forExpression);
		if (!string.IsNullOrWhiteSpace(aggregationTarget))
			AddNumberAggregation(aggregationTarget);
		instructions.Add(new LoopEndInstruction(instructions.Count - instructionCountBeforeLoopStart));
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

	private void GenerateInstructionForRangeLoopInstruction(MethodCall rangeExpression)
	{
		GenerateInstructionFromExpression(rangeExpression.Arguments[0]);
		var startIndexRegister = registry.PreviousRegister;
		GenerateInstructionFromExpression(rangeExpression.Arguments[1]);
		var endIndexRegister = registry.PreviousRegister;
		instructions.Add(new LoopBeginInstruction(startIndexRegister, endIndexRegister));
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
		instructions.Add(new JumpToId(InstructionType.JumpEnd, idStack.Pop()));
		if (ifExpression.OptionalElse == null)
			return;
		idStack.Push(conditionalId);
		instructions.Add(new JumpToId(InstructionType.JumpToIdIfTrue, conditionalId++));
		GenerateInstructions([ifExpression.OptionalElse]);
		instructions.Add(new JumpToId(InstructionType.JumpEnd, idStack.Pop()));
	}

	private void GenerateCodeForThen(If ifExpression)
	{
		if (ifExpression.Then is Body thenBody)
			GenerateInstructions(thenBody.Expressions);
		else
			GenerateInstructions([ifExpression.Then]);
	}

	private void GenerateCodeForBinary(MethodCall binary)
	{
		if (binary.Method.Name != "is")
			GenerateBinaryInstruction(binary,
				GetInstructionBasedOnBinaryOperationName(binary.Method.Name));
	}

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
		if (condition is Binary binary)
			GenerateForBinaryIfConditionalExpression(binary);
		else
			GenerateForBooleanCallIfCondition(condition);
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

	private void GenerateForBinaryIfConditionalExpression(Binary condition)
	{
		var leftRegister = GenerateLeftSideForIfCondition(condition);
		var rightRegister = GenerateRightSideForIfCondition(condition);
		GenerateInstructionsFromIfCondition(GetConditionalInstruction(condition.Method), leftRegister,
			rightRegister);
	}

	private void GenerateInstructionsFromIfCondition(InstructionType conditionInstruction,
		Register leftRegister, Register rightRegister)
	{
		instructions.Add(new BinaryInstruction(conditionInstruction, leftRegister, rightRegister));
		idStack.Push(conditionalId);
		instructions.Add(new JumpToId(InstructionType.JumpToIdIfFalse, conditionalId++));
	}

	private static InstructionType GetConditionalInstruction(Method condition) =>
		condition.Name switch
		{
			BinaryOperator.Greater => InstructionType.GreaterThan,
			BinaryOperator.Smaller => InstructionType.LessThan,
			_ => InstructionType.Equal
		};

	private Register GenerateRightSideForIfCondition(MethodCall condition)
	{
		GenerateInstructionFromExpression(condition.Arguments[0]);
		return registry.PreviousRegister;
	}

	private Register GenerateLeftSideForIfCondition(Binary condition) =>
		condition.Instance switch
		{
			Binary binaryInstance => GenerateValueBinaryInstructions(binaryInstance,
				GetInstructionBasedOnBinaryOperationName(binaryInstance.Method.Name)),
			MethodCall => InvokeAndGetStoredRegisterForConditional(condition),
			_ => LoadVariableForIfConditionLeft(condition)
		};

	private Register InvokeAndGetStoredRegisterForConditional(Binary condition)
	{
		if (condition.Instance == null)
			throw new InvalidOperationException(); //ncrunch: no coverage
		GenerateInstructionFromExpression(condition.Instance);
		return registry.PreviousRegister;
	}

	private Register LoadVariableForIfConditionLeft(Binary condition)
	{
		if (condition.Instance != null)
			GenerateInstructionFromExpression(condition.Instance);
		return registry.PreviousRegister;
	}

	private void GenerateBinaryInstruction(MethodCall binary, InstructionType operationInstruction)
	{
		if (binary.Instance is Binary binaryOp)
		{
			var leftReg = GenerateValueBinaryInstructions(binaryOp, operationInstruction);
			GenerateInstructionFromExpression(binary.Arguments[0]);
			instructions.Add(new BinaryInstruction(operationInstruction, leftReg, registry.PreviousRegister,
				registry.AllocateRegister()));
		}
		else if (binary.Arguments[0] is Binary binaryArg)
			GenerateNestedBinaryInstructions(binary, operationInstruction, binaryArg);
		else
			GenerateValueBinaryInstructions(binary, operationInstruction);
	}

	private void GenerateNestedBinaryInstructions(MethodCall binary,
		InstructionType operationInstruction, Binary binaryArgument)
	{
		var right = GenerateValueBinaryInstructions(binaryArgument,
			GetInstructionBasedOnBinaryOperationName(binaryArgument.Method.Name));
		var left = registry.AllocateRegister();
		if (binary.Instance != null)
			instructions.Add(new LoadVariableToRegister(left, binary.Instance.ToString()));
		instructions.Add(new BinaryInstruction(operationInstruction, left, right,
			registry.AllocateRegister()));
	}

	private Register GenerateValueBinaryInstructions(MethodCall binary,
		InstructionType operationInstruction)
	{
		if (binary.Instance == null)
			throw new InstanceNameNotFound(); //ncrunch: no coverage
		GenerateInstructionFromExpression(binary.Instance);
		var leftValue = registry.PreviousRegister;
		GenerateInstructionFromExpression(binary.Arguments[0]);
		var rightValue = registry.PreviousRegister;
		var resultRegister = registry.AllocateRegister();
		instructions.Add(new BinaryInstruction(operationInstruction, leftValue, rightValue,
			resultRegister));
		return resultRegister;
	}

	private sealed class InstanceNameNotFound : Exception;
}