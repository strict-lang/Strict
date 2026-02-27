using System.Collections;
using System.Diagnostics;
using Strict.Expressions;
using Strict.Language;
using Strict.Runtime.Statements;
using Return = Strict.Runtime.Statements.Return;
using Binary = Strict.Expressions.Binary;
using BinaryStatement = Strict.Runtime.Statements.Binary;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

public sealed class ByteCodeGenerator
{
	public ByteCodeGenerator(InvokedMethod method, Registry registry)
	{
		foreach (var argument in method.Arguments)
			statements.Add(new StoreVariableStatement(argument.Value, argument.Key));
		Expressions = method.Expressions;
		this.registry = registry;
		returnType = method.ReturnType;
		if (method is InstanceInvokedMethod instanceMethod)
			AddMembersFromCaller(instanceMethod.InstanceCall);
	}

	private readonly List<Statement> statements = [];
	private readonly Registry registry;
	private readonly Stack<int> idStack = new();
	private readonly Register[] registers = Enum.GetValues<Register>();
	private int conditionalId;

	public ByteCodeGenerator(MethodCall methodCall)
	{
		if (methodCall.Instance != null)
			AddInstanceMemberVariables((MethodCall)methodCall.Instance);
		AddMethodParameterVariables(methodCall);
		var methodBody = methodCall.Method.GetBodyAndParseIfNeeded();
		Expressions = methodBody is Body body
			? body.Expressions
			: [methodBody];
		registry = new Registry();
		returnType = methodCall.Method.ReturnType;
	}

	private IReadOnlyList<Expression> Expressions { get; }
	private readonly Type returnType;
	private int forResultId;

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
			if (instance.Method.Parameters[parameterIndex].Type is GenericTypeImplementation
				{
					Generic.Name: Base.List
				})
				statements.Add(new StoreVariableStatement(
					new Instance(instance.Method.Parameters[parameterIndex].Type, instance.Arguments),
					instance.ReturnType.Members[parameterIndex].Name));
			else
				statements.Add(new StoreVariableStatement(
					new Instance(instance.Arguments[parameterIndex], true),
					instance.ReturnType.Members[parameterIndex].Name));
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
			if ((expression.GetHashCode() == Expressions[^1].GetHashCode() ||
					expression is Expressions.Return) && expression is not If)
				GenerateStatementsFromReturn(expression);
			else
				GenerateStatementsFromExpression(expression);
		return statements;
	}

	private void GenerateStatementsFromReturn(Expression expression)
	{
		if (expression is Expressions.Return returnExpression)
		{
			if (!TryGenerateNumberForLoopReturn(returnExpression.Value))
			{
				GenerateStatementsFromExpression(returnExpression.Value);
				statements.Add(new Return(registry.PreviousRegister));
			}
			return;
		}
		if (TryGenerateNumberForLoopReturn(expression))
			return;
		GenerateStatementsFromExpression(expression);
		statements.Add(new Return(registry.PreviousRegister));
	}

	private bool TryGenerateNumberForLoopReturn(Expression expression)
	{
		if (expression is not For forExpression || !returnType.IsNumber)
			return false;
		GenerateStatementsForNumberAggregation(forExpression);
		return true;
	}

	private void GenerateStatementsForNumberAggregation(For forExpression)
	{
		var resultVariable = $"forResult{forResultId++}";
		statements.Add(new StoreVariableStatement(new Instance(returnType!, 0), resultVariable));
		GenerateLoopStatements(forExpression, resultVariable);
		statements.Add(new LoadVariableToRegister(registry.AllocateRegister(), resultVariable));
		statements.Add(new Return(registry.PreviousRegister));
	}

	private void GenerateStatementsFromExpression(Expression expression)
	{
		Debug.Assert(expression != null);
		_ = TryGenerateBodyStatements(expression) ?? TryGenerateBinaryStatements(expression) ??
			TryGenerateIfStatements(expression) ?? TryGenerateAssignmentStatements(expression) ??
			TryGenerateLoopStatements(expression) ?? TryGenerateMutableStatements(expression) ??
			TryGenerateMemberCallStatement(expression) ??
			TryGenerateVariableCallStatement(expression) ?? TryGenerateValueStatement(expression) ??
			TryGenerateMethodCallStatement(expression) ?? TryGenerateListCallStatement(expression) ??
			throw new NotSupportedException(expression.ToString());
	}

	private bool? TryGenerateListCallStatement(Expression expression)
	{
		if (expression is not ListCall listCall)
			return null; //ncrunch: no coverage
		GenerateStatementsFromExpression(listCall.Index);
		var indexRegister = registry.PreviousRegister;
		statements.Add(new ListCallStatement(registry.AllocateRegister(), indexRegister,
			listCall.List.ToString()));
		return true;
	}

	private bool? TryGenerateVariableCallStatement(Expression expression)
	{
		if (expression is not (VariableCall or ParameterCall))
			return null;
		statements.Add(
			new LoadVariableToRegister(registry.AllocateRegister(), expression.ToString()));
		return true;
	}

	private bool? TryGenerateMemberCallStatement(Expression expression)
	{
		if (expression is not MemberCall memberCall)
			return null;
		if (memberCall.Instance == null)
			statements.Add(
				new LoadVariableToRegister(registry.AllocateRegister(), expression.ToString()));
		else if (memberCall.Member.InitialValue != null)
			TryGenerateForEnum(memberCall.Instance.ReturnType, memberCall.Member.InitialValue);
		return true;
	}

	private void TryGenerateForEnum(Type type, Expression value)
	{
		if (type.IsEnum)
			statements.Add(new LoadConstantStatement(registry.AllocateRegister(),
				new Instance(type, value)));
	}

	private bool? TryGenerateValueStatement(Expression expression)
	{
		if (expression is not Value valueExpression)
			return null;
		statements.Add(new LoadConstantStatement(registry.AllocateRegister(),
			new Instance(valueExpression.ReturnType, valueExpression.Data)));
		return true;
	}

	private bool? TryGenerateMethodCallStatement(Expression expression)
	{
		if (expression is Binary || expression is not MethodCall methodCall)
			return null;
		if (TryGenerateStatementForCollectionManipulation(methodCall))
			return true;
		statements.Add(new Invoke(registry.AllocateRegister(), methodCall, registry));
		return true;
	}

	private bool TryGenerateStatementForCollectionManipulation(MethodCall methodCall)
	{
		switch (methodCall.Method.Name)
		{
		case "Add":
		{
			GenerateStatementsForAddMethod(methodCall);
			return true;
		}
		case "Remove":
		{
			GenerateStatementsForRemoveMethod(methodCall);
			return true;
		}
		case "Increment":
		case "Decrement":
		{
			var register = registry.AllocateRegister();
			statements.Add(new Invoke(register, methodCall, registry));
			if (methodCall.Instance != null)
				statements.Add(new StoreFromRegisterStatement(register, methodCall.Instance.ToString()));
			return true;
		}
		default:
			return false;
		}
	}

	private void GenerateStatementsForRemoveMethod(MethodCall methodCall)
	{
		if (methodCall.Instance == null)
			return; //ncrunch: no coverage
		GenerateStatementsFromExpression(methodCall.Arguments[0]);
		if (methodCall.Instance.ReturnType is GenericTypeImplementation { Generic.Name: Base.List })
			statements.Add(new RemoveStatement(methodCall.Instance.ToString(),
				registry.PreviousRegister));
	}

	private void GenerateStatementsForAddMethod(MethodCall methodCall)
	{
		if (TryGenerateAddForTable(methodCall) || methodCall.Instance == null)
			return;
		GenerateStatementsFromExpression(methodCall.Arguments[0]);
		statements.Add(new WriteToListStatement(registry.PreviousRegister,
			methodCall.Instance.ToString()));
	}

	private bool TryGenerateAddForTable(MethodCall methodCall)
	{
		if (methodCall.Arguments.Count != 2 || methodCall.Instance == null)
			return false;
		GenerateStatementsFromExpression(methodCall.Arguments[0]);
		var key = registry.PreviousRegister;
		GenerateStatementsFromExpression(methodCall.Arguments[1]);
		var value = registry.PreviousRegister;
		statements.Add(new WriteToTableStatement(key, value, methodCall.Instance.ToString()));
		return true;
	}

	private bool? TryGenerateBodyStatements(Expression expression)
	{
		if (expression is not Body body)
			return null;
		GenerateStatements(body.Expressions);
		return true;
	}

	private bool? TryGenerateMutableStatements(Expression expression)
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
			TryGenerateStatementsForAssignmentValue(declarationOrAssignmentValue, name);
		else
		{
			GenerateStatementsFromExpression(declarationOrAssignment);
			statements.Add(new StoreFromRegisterStatement(registers[registry.NextRegister - 1], name));
		}
	}

	private bool? TryGenerateLoopStatements(Expression expression)
	{
		if (expression is not For forExpression)
			return null;
		GenerateLoopStatements(forExpression);
		return true;
	}

	private bool? TryGenerateAssignmentStatements(Expression expression)
	{
		if (expression is not Declaration assignmentExpression || expression.IsMutable)
			return null;
		GenerateForAssignmentOrDeclaration(assignmentExpression.Value, assignmentExpression.Name);
		return true;
	}

	private void TryGenerateStatementsForAssignmentValue(Value assignmentValue, string variableName)
	{
		var data = assignmentValue.Data switch
		{
			IEnumerable<Expression> expressions => expressions.ToList(),
			IDictionary => new Dictionary<Value, Value>(),
			_ => assignmentValue.Data
		};
		statements.Add(new StoreVariableStatement(new Instance(assignmentValue.ReturnType, data),
			variableName));
	}

	private bool? TryGenerateIfStatements(Expression expression)
	{
		if (expression is not If ifExpression)
			return null;
		GenerateIfStatements(ifExpression);
		return true;
	}

	private bool? TryGenerateBinaryStatements(Expression expression)
	{
		if (expression is not Binary binary)
			return null;
		GenerateCodeForBinary(binary);
		return true;
	}

	private void GenerateLoopStatements(For forExpression, string? aggregationTarget = null)
	{
		var statementCountBeforeLoopStart = statements.Count;
		if (forExpression.Iterator is MethodCall rangeExpression &&
			forExpression.Iterator.ReturnType.Name == Base.Range &&
			rangeExpression.Method.Name == Method.From)
			GenerateStatementForRangeLoopInstruction(rangeExpression);
		else
		{
			GenerateStatementsFromExpression(forExpression.Iterator);
			statements.Add(new LoopBeginStatement(registry.PreviousRegister));
		}
		GenerateStatementsForLoopBody(forExpression);
		if (!string.IsNullOrWhiteSpace(aggregationTarget))
			AddNumberAggregation(aggregationTarget);
		statements.Add(new LoopEndStatement(statements.Count - statementCountBeforeLoopStart));
	}

	private void AddNumberAggregation(string aggregationTarget)
	{
		var loopValueRegister = registry.PreviousRegister;
		statements.Add(new LoadVariableToRegister(registry.AllocateRegister(), aggregationTarget));
		var accumulatorRegister = registry.PreviousRegister;
		statements.Add(new BinaryStatement(Instruction.Add, accumulatorRegister, loopValueRegister,
			registry.AllocateRegister()));
		statements.Add(new StoreFromRegisterStatement(registry.PreviousRegister, aggregationTarget));
	}

	private void GenerateStatementForRangeLoopInstruction(MethodCall rangeExpression)
	{
		GenerateStatementsFromExpression(rangeExpression.Arguments[0]);
		var startIndexRegister = registry.PreviousRegister;
		GenerateStatementsFromExpression(rangeExpression.Arguments[1]);
		var endIndexRegister = registry.PreviousRegister;
		statements.Add(new LoopBeginStatement(startIndexRegister, endIndexRegister));
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
		GenerateCodeForIfCondition(ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
		statements.Add(new JumpToId(Instruction.JumpEnd, idStack.Pop()));
		if (ifExpression.OptionalElse == null)
			return;
		idStack.Push(conditionalId);
		statements.Add(new JumpToId(Instruction.JumpToIdIfTrue, conditionalId++));
		GenerateStatements([ifExpression.OptionalElse]);
		statements.Add(new JumpToId(Instruction.JumpEnd, idStack.Pop()));
	}

	private void GenerateCodeForThen(If ifExpression)
	{
		if (ifExpression.Then is Body thenBody)
			GenerateStatements(thenBody.Expressions);
		else
			GenerateStatements([ifExpression.Then]);
	}

	private void GenerateCodeForBinary(MethodCall binary)
	{
		if (binary.Method.Name != "is")
			GenerateBinaryStatement(binary,
				GetInstructionBasedOnBinaryOperationName(binary.Method.Name));
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

	private void GenerateCodeForIfCondition(Expression condition)
	{
		if (condition is Binary binary)
			GenerateForBinaryIfConditionalExpression(binary);
		else
			GenerateForBooleanCallIfCondition(condition);
	}

	private void GenerateForBooleanCallIfCondition(Expression condition)
	{
		GenerateStatementsFromExpression(condition);
		var instanceCallRegister = registry.PreviousRegister;
		statements.Add(new LoadConstantStatement(registry.AllocateRegister(),
			new Instance(condition.ReturnType, true)));
		GenerateInstructionsFromIfCondition(Instruction.Equal, instanceCallRegister,
			registry.PreviousRegister);
	}

	private void GenerateForBinaryIfConditionalExpression(Binary condition)
	{
		var leftRegister = GenerateLeftSideForIfCondition(condition);
		var rightRegister = GenerateRightSideForIfCondition(condition);
		GenerateInstructionsFromIfCondition(GetConditionalInstruction(condition.Method), leftRegister,
			rightRegister);
	}

	private void GenerateInstructionsFromIfCondition(Instruction conditionInstruction,
		Register leftRegister, Register rightRegister)
	{
		statements.Add(new BinaryStatement(conditionInstruction, leftRegister, rightRegister));
		idStack.Push(conditionalId);
		statements.Add(new JumpToId(Instruction.JumpToIdIfFalse, conditionalId++));
	}

	private static Instruction GetConditionalInstruction(Method condition) =>
		condition.Name switch
		{
			BinaryOperator.Greater => Instruction.GreaterThan,
			BinaryOperator.Smaller => Instruction.LessThan,
			_ => Instruction.Equal
		};

	private Register GenerateRightSideForIfCondition(MethodCall condition)
	{
		GenerateStatementsFromExpression(condition.Arguments[0]);
		return registry.PreviousRegister;
	}

	private Register GenerateLeftSideForIfCondition(Binary condition) =>
		condition.Instance switch
		{
			Binary binaryInstance => GenerateValueBinaryStatements(binaryInstance,
				GetInstructionBasedOnBinaryOperationName(binaryInstance.Method.Name)),
			MethodCall => InvokeAndGetStoredRegisterForConditional(condition),
			_ => LoadVariableForIfConditionLeft(condition)
		};

	private Register InvokeAndGetStoredRegisterForConditional(Binary condition)
	{
		if (condition.Instance == null)
			throw new InvalidOperationException(); //ncrunch: no coverage
		GenerateStatementsFromExpression(condition.Instance);
		return registry.PreviousRegister;
	}

	private Register LoadVariableForIfConditionLeft(Binary condition)
	{
		if (condition.Instance != null)
			GenerateStatementsFromExpression(condition.Instance);
		return registry.PreviousRegister;
	}

	private void GenerateBinaryStatement(MethodCall binary, Instruction operationInstruction)
	{
		if (binary.Instance is Binary binaryOp)
		{
			var leftReg = GenerateValueBinaryStatements(binaryOp, operationInstruction);
			GenerateStatementsFromExpression(binary.Arguments[0]);
			statements.Add(new BinaryStatement(operationInstruction, leftReg, registry.PreviousRegister,
				registry.AllocateRegister()));
		}
		else if (binary.Arguments[0] is Binary binaryArg)
			GenerateNestedBinaryStatements(binary, operationInstruction, binaryArg);
		else
			GenerateValueBinaryStatements(binary, operationInstruction);
	}

	private void GenerateNestedBinaryStatements(MethodCall binary, Instruction operationInstruction,
		Binary binaryArgument)
	{
		var right = GenerateValueBinaryStatements(binaryArgument,
			GetInstructionBasedOnBinaryOperationName(binaryArgument.Method.Name));
		var left = registry.AllocateRegister();
		if (binary.Instance != null)
			statements.Add(new LoadVariableToRegister(left, binary.Instance.ToString()));
		statements.Add(new BinaryStatement(operationInstruction, left, right,
			registry.AllocateRegister()));
	}

	private Register GenerateValueBinaryStatements(MethodCall binary,
		Instruction operationInstruction)
	{
		if (binary.Instance == null)
			throw new InstanceNameNotFound(); //ncrunch: no coverage
		GenerateStatementsFromExpression(binary.Instance);
		var leftValue = registry.PreviousRegister;
		GenerateStatementsFromExpression(binary.Arguments[0]);
		var rightValue = registry.PreviousRegister;
		var resultRegister = registry.AllocateRegister();
		statements.Add(new BinaryStatement(operationInstruction, leftValue, rightValue,
			resultRegister));
		return resultRegister;
	}

	private sealed class InstanceNameNotFound : Exception;
}