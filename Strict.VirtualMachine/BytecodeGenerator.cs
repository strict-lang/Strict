using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class ByteCodeGenerator
{
	private readonly Register[] registers = Enum.GetValues<Register>();
	private readonly List<Statement> statements = new();
	private int conditionalId;
	private readonly Stack<int> idStack = new();
	private readonly Registry registry;

	public ByteCodeGenerator(InvokedMethod method, Registry registry)
	{
		foreach (var argument in method.Arguments)
			statements.Add(new StoreVariableStatement(argument.Value, argument.Key));
		Expressions = method.Expressions;
		this.registry = registry;
		if (method is not InstanceInvokedMethod instanceMethod)
			return;
		AddMembersFromCaller(instanceMethod.InstanceCall);
	}

	public ByteCodeGenerator(MethodCall methodCall)
	{
		if (methodCall.Instance != null)
			AddInstanceMemberVariables((MethodCall)methodCall.Instance);
		AddMethodParameterVariables(methodCall);
		Expressions = ((Body)methodCall.Method.GetBodyAndParseIfNeeded()).Expressions;
		registry = new Registry();
	}

	private IReadOnlyList<Expression> Expressions { get; }

	private void AddMembersFromCaller(Instance instance)
	{
		if (instance.ReturnType != null)
			statements.Add(new StoreVariableStatement(instance,
				instance.ReturnType.Members.First(member => !member.Type.IsTrait).Name));
	}

	private void AddInstanceMemberVariables(MethodCall instance)
	{
		for (var parameterIndex = 0; parameterIndex < instance.Method.Parameters.Count;
			parameterIndex++)
		{
			if (instance.Method.Parameters[parameterIndex].Type is GenericTypeImplementation type &&
				type.Generic.Name == Base.List)
				statements.Add(new StoreVariableStatement(
					new Instance(instance.Method.Parameters[parameterIndex].Type,
						instance.Arguments[parameterIndex]),
					instance.ReturnType.Members[parameterIndex].Name));
			else
				statements.Add(new StoreVariableStatement(
					new Instance(instance.Arguments[parameterIndex], true),
					instance.ReturnType.Members[parameterIndex].Name));
			if (instance.ReturnType.Members[parameterIndex].Value == null) //TODO: This condition is necessary for now for test synchronization
				instance.ReturnType.Members[parameterIndex].
					UpdateValue(instance.Arguments[parameterIndex],
						new Body(instance.Method));
		}
	}

	private void AddMethodParameterVariables(MethodCall methodCall)
	{
		for (var parameterIndex = 0; parameterIndex < methodCall.Method.Parameters.Count;
			parameterIndex++)
			statements?.Add(new StoreVariableStatement(
				new Instance(methodCall.Arguments[parameterIndex]),
				methodCall.Method.Parameters[parameterIndex].Name));
	}

	public List<Statement> Generate() => GenerateStatements(Expressions);

	private List<Statement> GenerateStatements(IEnumerable<Expression> expressions)
	{
		foreach (var expression in expressions)
			if ((expression.GetHashCode() == Expressions[^1].GetHashCode() || expression is Return) &&
				expression is not If)
				GenerateStatementsFromReturn(expression);
			else
				GenerateStatementsFromExpression(expression);
		return statements;
	}

	private void GenerateStatementsFromReturn(Expression expression)
	{
		if (expression is Return returnExpression)
			GenerateStatementsFromExpression(returnExpression.Value);
		else
			GenerateStatementsFromExpression(expression);
		statements.Add(new ReturnStatement(registry.PreviousRegister));
	}

	private void GenerateStatementsFromExpression(Expression expression)
	{
		TryGenerateBodyStatements(expression);
		TryGenerateBinaryStatements(expression);
		TryGenerateIfStatements(expression);
		TryGenerateAssignmentStatements(expression);
		TryGenerateLoopStatements(expression);
		TryGenerateMutableStatements(expression);
		TryGenerateVariableOrMemberCallStatement(expression);
		TryGenerateMethodCallStatement(expression);
		TryGenerateValueStatement(expression);
	}

	private void TryGenerateValueStatement(Expression expression)
	{
		if (expression is Value valueExpression)
			statements.Add(new LoadConstantStatement(registry.AllocateRegister(),
				new Instance(valueExpression.ReturnType, valueExpression.Data)));
	}

	private void TryGenerateMethodCallStatement(Expression expression)
	{
		if (expression is Binary || expression is not MethodCall methodCall)
			return;
		if (methodCall.Method.Name == "Add" && methodCall.Instance != null)
		{
			GenerateStatementsFromExpression(methodCall.Arguments[0]);
			statements.Add(new WriteToListStatement(registry.PreviousRegister,
				methodCall.Instance.ToString()));
		}
		else
			statements.Add(new InvokeStatement(methodCall, registry.AllocateRegister(), registry));
	}

	private void TryGenerateBodyStatements(Expression expression)
	{
		if (expression is Body body)
			GenerateStatements(body.Expressions);
	}

	private void TryGenerateVariableOrMemberCallStatement(Expression expression)
	{
		if (expression is VariableCall or MemberCall)
			statements.Add(new LoadVariableStatement(registry.AllocateRegister(), expression.ToString()));
	}

	private void TryGenerateMutableStatements(Expression expression)
	{
		if (expression is MutableDeclaration declaration)
			GenerateForAssignmentOrDeclaration(declaration.Value, declaration.Name);
		else if (expression is MutableAssignment assignment)
			GenerateForAssignmentOrDeclaration(assignment.Value, assignment.Name);
	}

	private void GenerateForAssignmentOrDeclaration(Expression declarationOrAssignment, string name)
	{
		if (declarationOrAssignment is List listDeclaration)
		{
			statements.Add(new StoreVariableStatement(
				new Instance(declarationOrAssignment.ReturnType, listDeclaration.Values), name));
		}
		else if (declarationOrAssignment is Value declarationOrAssignmentValue)
		{
			TryGenerateStatementsForAssignmentValue(declarationOrAssignmentValue, name);
		}
		else
		{
			GenerateStatementsFromExpression(declarationOrAssignment);
			statements.Add(new StoreFromRegisterStatement(registers[registry.NextRegister - 1], name));
		}
	}

	private void TryGenerateLoopStatements(Expression expression)
	{
		if (expression is For forExpression)
			GenerateLoopStatements(forExpression);
	}

	private void TryGenerateAssignmentStatements(Expression expression)
	{
		if (expression is not ConstantDeclaration assignmentExpression || expression.IsMutable)
			return;
		GenerateForAssignmentOrDeclaration(assignmentExpression.Value, assignmentExpression.Name);
	}

	private void
		TryGenerateStatementsForAssignmentValue(Value assignmentValue, string variableName) =>
		statements.Add(new StoreVariableStatement(
			new Instance(assignmentValue.ReturnType, assignmentValue.Data), variableName));

	private void TryGenerateIfStatements(Expression expression)
	{
		if (expression is If ifExpression)
			GenerateIfStatements(ifExpression);
	}

	private void TryGenerateBinaryStatements(Expression expression)
	{
		if (expression is Binary binary)
			GenerateCodeForBinary(binary);
	}

	private void GenerateLoopStatements(For forExpression)
	{
		var statementCountBeforeLoopStart = statements.Count;
		statements.Add(new LoopBeginStatement(forExpression.Value.ToString()));
		GenerateStatementsForLoopBody(forExpression);
		statements.Add(new IterationEndStatement(statements.Count - statementCountBeforeLoopStart));
	}

	private void GenerateStatementsForLoopBody(For forExpression)
	{
		if (forExpression.Body is Body forExpressionBody)
			GenerateStatements(forExpressionBody.Expressions);
		else
			GenerateStatementsFromExpression(forExpression.Body);
	}

	private void GenerateIfStatements(If ifExpression)
	{
		GenerateCodeForIfCondition((Binary)ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
		statements.Add(new JumpToIdStatement(Instruction.JumpEnd, idStack.Pop()));
		if (ifExpression.OptionalElse == null)
			return;
		idStack.Push(conditionalId);
		statements.Add(new JumpToIdStatement(Instruction.JumpToIdIfTrue, conditionalId++));
		GenerateStatements(new[] { ifExpression.OptionalElse });
		statements.Add(new JumpToIdStatement(Instruction.JumpEnd, idStack.Pop()));
	}

	private void GenerateCodeForThen(If ifExpression)
	{
		if (ifExpression.Then is Body thenBody)
			GenerateStatements(thenBody.Expressions);
		else
			GenerateStatements(new[] { ifExpression.Then });
	}

	private void GenerateCodeForBinary(MethodCall binary)
	{
		if (binary.Method.Name != "is")
		{
			var instruction = GetInstructionBasedOnBinaryOperationName(binary.Method.Name);
			GenerateBinaryStatement(binary, instruction);
		}
	}

	private static Instruction GetInstructionBasedOnBinaryOperationName(string binaryOperator) =>
		binaryOperator switch
		{
			BinaryOperator.Plus => Instruction.Add,
			BinaryOperator.Multiply => Instruction.Multiply,
			BinaryOperator.Minus => Instruction.Subtract,
			BinaryOperator.Divide => Instruction.Divide,
			BinaryOperator.Modulate => Instruction.Modulo,
			_ => throw new NotImplementedException() //ncrunch: no coverage
		};

	private void GenerateCodeForIfCondition(Binary condition)
	{
		var leftRegister = GenerateLeftSideForIfCondition(condition);
		var rightRegister = GenerateRightSideForIfCondition(condition);
		statements.Add(new BinaryStatement(GetConditionalInstruction(condition.Method), leftRegister,
			rightRegister));
		idStack.Push(conditionalId);
		statements.Add(new JumpToIdStatement(Instruction.JumpToIdIfFalse, conditionalId++));
	}

	private static Instruction GetConditionalInstruction(Method condition) =>
		condition.Name switch
		{
			BinaryOperator.Greater => Instruction.GreaterThan,
			BinaryOperator.Smaller => Instruction.LessThan, //ncrunch: no coverage
			BinaryOperator.Is => Instruction.Equal,
			_ => Instruction.Equal //ncrunch: no coverage
		};

	private Register GenerateRightSideForIfCondition(MethodCall condition)
	{
		var rightRegister = registry.AllocateRegister();
		if (condition.Arguments[0] is Value argumentValue)
			statements.Add(new LoadConstantStatement(rightRegister,
				new Instance(argumentValue.ReturnType, argumentValue.Data)));
		else
			statements.Add(new LoadVariableStatement(rightRegister, condition.Arguments[0].ToString())); //ncrunch: no coverage
		return rightRegister;
	}

	private Register GenerateLeftSideForIfCondition(Binary condition) =>
		condition.Instance switch
		{
			Binary binaryInstance => GenerateValueBinaryStatements(binaryInstance,
				GetInstructionBasedOnBinaryOperationName(binaryInstance.Method.Name)),
			MethodCall => InvokeAndGetStoredRegisterForConditional(condition), //ncrunch: no coverage, TODO: missing tests
			_ => LoadVariableForIfConditionLeft(condition)
		};

	//ncrunch: no coverage start, TODO: missing tests
	private Register InvokeAndGetStoredRegisterForConditional(Binary condition)
	{
		if (condition.Instance == null)
			throw new InvalidOperationException(); //ncrunch: no coverage
		GenerateStatementsFromExpression(condition.Instance);
		return registry.PreviousRegister;
	} //ncrunch: no coverage end

	private Register LoadVariableForIfConditionLeft(Binary condition)
	{
		var leftRegister = registry.AllocateRegister();
		statements.Add(new LoadVariableStatement(leftRegister,
			condition.Instance?.ToString() ?? throw new InvalidOperationException()));
		return leftRegister;
	}

	private void GenerateBinaryStatement(MethodCall binary, Instruction operationInstruction)
	{
		if (binary.Instance is Binary binaryOp)
		{
			var left = GenerateValueBinaryStatements(binaryOp, operationInstruction);
			statements.Add(new BinaryStatement(operationInstruction, left, registry.AllocateRegister(),
				registry.AllocateRegister()));
		}
		else if (binary.Arguments[0] is Binary binaryArg)
		{
			GenerateNestedBinaryStatements(binary, operationInstruction, binaryArg);
		}
		else
		{
			GenerateValueBinaryStatements(binary, operationInstruction);
		}
	}

	private void GenerateNestedBinaryStatements(MethodCall binary, Instruction operationInstruction,
		Binary binaryArg)
	{
		var right = GenerateValueBinaryStatements(binaryArg,
			GetInstructionBasedOnBinaryOperationName(binaryArg.Method.Name));
		var left = registry.AllocateRegister();
		if (binary.Instance != null)
			statements.Add(new LoadVariableStatement(left, binary.Instance.ToString()));
		statements.Add(new BinaryStatement(operationInstruction, left, right, registry.AllocateRegister()));
	}

	private Register GenerateValueBinaryStatements(MethodCall binary,
		Instruction operationInstruction)
	{
		var (leftRegister, rightRegister, resultRegister) =
			(registry.AllocateRegister(), registry.AllocateRegister(), registry.AllocateRegister());
		if (binary.Instance is Value instanceValue)
			statements.Add(new LoadConstantStatement(leftRegister, new Instance(instanceValue)));
		else
			statements.Add(new LoadVariableStatement(leftRegister,
				binary.Instance?.ToString() ?? throw new InstanceNameNotFound()));
		if (binary.Arguments[0] is Value argumentsValue)
			statements.Add(new LoadConstantStatement(rightRegister, new Instance(argumentsValue)));
		else
			statements.Add(new LoadVariableStatement(rightRegister, binary.Arguments[0].ToString()));
		statements.Add(new BinaryStatement(operationInstruction, leftRegister, rightRegister,
			resultRegister));
		return resultRegister;
	}

	private sealed class InstanceNameNotFound : Exception { }
}