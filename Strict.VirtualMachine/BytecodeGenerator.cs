﻿using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

// ReSharper disable once ClassTooBig
public sealed class ByteCodeGenerator
{
	private readonly Register[] registers = Enum.GetValues<Register>();
	private readonly List<Statement> statements = new();
	private int conditionalId;
	private readonly Stack<int> idStack = new();
	private int nextRegister;
	private Register previousRegister;

	public ByteCodeGenerator(MethodCall methodCall)
	{
		InstanceArguments = new Dictionary<string, Expression>();
		Method = methodCall.Method;
		if (methodCall.Instance != null)
			AddInstanceMemberVariables((MethodCall)methodCall.Instance);
		AddMethodParameterVariables(methodCall);
		StoreAndLoadVariables();
		Expressions = ((Body)Method.GetBodyAndParseIfNeeded()).Expressions;
	}

	public IReadOnlyList<Expression> Expressions { get; }
	private Dictionary<string, Expression>? InstanceArguments { get; }
	private Method Method { get; }

	private void StoreAndLoadVariables()
	{
		if (InstanceArguments != null)
			foreach (var argument in InstanceArguments)
				statements.Add(new StoreStatement(new Instance(argument.Value), argument.Key));
	}

	private void AddInstanceMemberVariables(MethodCall instance)
	{
		for (var parameterIndex = 0; parameterIndex < instance.Method.Parameters.Count;
			parameterIndex++)
			if (instance.Method.Parameters[parameterIndex].Type is GenericTypeImplementation type &&
				type.Generic.Name == Base.List)
				statements.Add(new StoreStatement(
					new Instance(instance.Method.Parameters[parameterIndex].Type,
						instance.Arguments[parameterIndex]),
					instance.ReturnType.Members[parameterIndex].Name));
			else
				InstanceArguments?.Add(instance.ReturnType.Members[parameterIndex].Name,
					instance.Arguments[parameterIndex]);
	}

	private void AddMethodParameterVariables(MethodCall methodCall)
	{
		for (var parameterIndex = 0; parameterIndex < Method.Parameters.Count; parameterIndex++)
			InstanceArguments?.Add(Method.Parameters[parameterIndex].Name,
				methodCall.Arguments[parameterIndex]);
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
		statements.Add(new ReturnStatement(previousRegister));
	}

	private void GenerateStatementsFromExpression(Expression expression)
	{
		TryGenerateBodyStatements(expression);
		TryGenerateBinaryStatements(expression);
		TryGenerateIfStatements(expression);
		TryGenerateAssignmentStatements(expression);
		TryGenerateLoopStatements(expression);
		TryGenerateMutableStatements(expression);
		TryGenerateVariableCallStatement(expression);
	}

	private void TryGenerateBodyStatements(Expression expression)
	{
		if (expression is Body body)
			GenerateStatements(body.Expressions);
	}

	private void TryGenerateVariableCallStatement(Expression expression)
	{
		if (expression is VariableCall)
			statements.Add(new LoadVariableStatement(AllocateRegister(), expression.ToString()));
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
		if (declarationOrAssignment is Value declarationOrAssignmentValue)
		{
			TryGenerateStatementsForAssignmentValue(declarationOrAssignmentValue, name);
		}
		else
		{
			GenerateStatementsFromExpression(declarationOrAssignment);
			statements.Add(new StoreFromRegisterStatement(registers[nextRegister - 1], name));
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
		statements.Add(new StoreStatement(
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
		var iterableName = forExpression.Value.ToString();
		var iterableInstance = statements.LastOrDefault(statement =>
				statement is StoreStatement storeStatement && storeStatement.Identifier == iterableName)?.
			Instance;
		if (iterableInstance != null)
			GenerateRestLoopStatements(forExpression, iterableInstance);
		FreeRegisters();
	}

	private static int? GetLength(Instance iterableInstance)
	{
		if (iterableInstance.Value is string iterableString)
			return iterableString.Length;
		if (iterableInstance.Value is int or double)
			return Convert.ToInt32(iterableInstance.Value);
		if (iterableInstance.ReturnType != null && iterableInstance.ReturnType.IsIterator)
			return ((IEnumerable<Expression>)iterableInstance.Value).Count();
		return 0; //ncrunch: no coverage
	}

	private void GenerateRestLoopStatements(For forExpression, Instance iterableInstance)
	{
		var length = GetLength(iterableInstance);
		if (length == null)
			return; //ncrunch: no coverage
		var (registerForIterationCount, registerForIndexReduction) =
			(AllocateRegister(true), AllocateRegister(true));
		statements.Add(new LoadConstantStatement(registerForIterationCount,
			new Instance(iterableInstance.ReturnType, length)));
		statements.Add(new LoadConstantStatement(registerForIndexReduction,
			new Instance(iterableInstance.ReturnType, 1)));
		var statementCountBeforeLoopStart = statements.Count;
		statements.Add(new InitLoopStatement(forExpression.Value.ToString()));
		GenerateStatementsForLoopBody(forExpression);
		GenerateIteratorReductionAndJumpStatementsForLoop(registerForIterationCount,
			registerForIndexReduction, statements.Count - statementCountBeforeLoopStart);
	}

	private void GenerateStatementsForLoopBody(For forExpression)
	{
		if (forExpression.Body is Body forExpressionBody)
			GenerateStatements(forExpressionBody.Expressions);
		else
			GenerateStatementsFromExpression(forExpression.Body);
	}

	private void GenerateIteratorReductionAndJumpStatementsForLoop(
		Register registerForIterationCount, Register registerForIndexReduction, int steps)
	{
		statements.Add(new Statement(Instruction.Subtract, registerForIterationCount,
			registerForIndexReduction, registerForIterationCount));
		statements.Add(new JumpStatement(Instruction.JumpIfNotZero, -steps - 2,
			registerForIterationCount));
	}

	private void GenerateIfStatements(If ifExpression)
	{
		GenerateCodeForIfCondition((Binary)ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
		statements.Add(new JumpViaIdStatement(Instruction.JumpEnd, idStack.Pop()));
		if (ifExpression.OptionalElse == null)
			return;
		idStack.Push(conditionalId);
		statements.Add(new JumpViaIdStatement(Instruction.JumpToIdIfTrue, conditionalId++));
		GenerateStatements(new[] { ifExpression.OptionalElse });
		statements.Add(new JumpViaIdStatement(Instruction.JumpEnd, idStack.Pop()));
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
		statements.Add(new Statement(GetConditionalInstruction(condition.Method), leftRegister,
			rightRegister));
		idStack.Push(conditionalId);
		statements.Add(new JumpViaIdStatement(Instruction.JumpToIdIfFalse, conditionalId++));
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
		var rightRegister = AllocateRegister();
		if (condition.Arguments[0] is Value argumentValue)
			statements.Add(new LoadConstantStatement(rightRegister,
				new Instance(argumentValue.ReturnType, argumentValue.Data)));
		else
			statements.Add(new LoadVariableStatement(rightRegister, condition.Arguments[0].ToString())); //ncrunch: no coverage
		return rightRegister;
	}

	private Register GenerateLeftSideForIfCondition(MethodCall condition) =>
		condition.Instance switch
		{
			Value instanceValue => LoadConstantForIfConditionLeft(instanceValue), //ncrunch: no coverage
			Binary binaryInstance => GenerateValueBinaryStatements(binaryInstance,
				GetInstructionBasedOnBinaryOperationName(binaryInstance.Method.Name)),
			_ => LoadVariableForIfConditionLeft(condition)
		};

	//ncrunch: no coverage start
	private Register LoadConstantForIfConditionLeft(Value instanceValue)
	{
		var leftRegister = AllocateRegister();
		statements.Add(new LoadConstantStatement(leftRegister,
			new Instance(instanceValue.ReturnType, instanceValue.Data)));
		return leftRegister;
	} //ncrunch: no coverage end

	private Register LoadVariableForIfConditionLeft(MethodCall condition)
	{
		var leftRegister = AllocateRegister();
		statements.Add(new LoadVariableStatement(leftRegister,
			condition.Instance?.ToString() ?? throw new InvalidOperationException()));
		return leftRegister;
	}

	private Register AllocateRegister(bool isLocked = false)
	{
		if (nextRegister == registers.Length)
			nextRegister = 0;
		previousRegister = registers[nextRegister];
		var currentRegister = registers[nextRegister++];
		if (lockedRegisters.Contains(currentRegister))
			currentRegister = AllocateRegister();
		if (isLocked)
			lockedRegisters.Add(currentRegister);
		return currentRegister;
	}

	private void FreeRegisters() => lockedRegisters.Clear();
	private readonly List<Register> lockedRegisters = new();

	private void GenerateBinaryStatement(MethodCall binary, Instruction operationInstruction)
	{
		if (binary.Instance is Binary binaryOp)
		{
			var left = GenerateValueBinaryStatements(binaryOp, operationInstruction);
			statements.Add(new Statement(operationInstruction, left, AllocateRegister(),
				AllocateRegister()));
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
		var left = AllocateRegister();
		if (binary.Instance != null)
			statements.Add(new LoadVariableStatement(left, binary.Instance.ToString()));
		statements.Add(new Statement(operationInstruction, left, right, AllocateRegister()));
	}

	private Register GenerateValueBinaryStatements(MethodCall binary,
		Instruction operationInstruction)
	{
		var (leftRegister, rightRegister, resultRegister) =
			(AllocateRegister(), AllocateRegister(), AllocateRegister());
		if (binary.Instance is Value instanceValue)
			statements.Add(new LoadConstantStatement(leftRegister, new Instance(instanceValue)));
		else
			statements.Add(new LoadVariableStatement(leftRegister,
				binary.Instance?.ToString() ?? throw new InstanceNameNotFound()));
		if (binary.Arguments[0] is Value argumentsValue)
			statements.Add(new LoadConstantStatement(rightRegister, new Instance(argumentsValue)));
		else
			statements.Add(new LoadVariableStatement(rightRegister, binary.Arguments[0].ToString()));
		statements.Add(new Statement(operationInstruction, leftRegister, rightRegister,
			resultRegister));
		return resultRegister;
	}

	private sealed class InstanceNameNotFound : Exception { }
}