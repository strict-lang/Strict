using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

public sealed class ByteCodeGenerator
{
	private readonly Stack<int> idStack = new();
	private readonly Register[] registers = Enum.GetValues<Register>();
	private readonly Registry registry;
	private readonly List<Statement> statements = new();
	private int conditionalId;

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
		var methodBody = methodCall.Method.GetBodyAndParseIfNeeded();
		Expressions = methodBody is not Body
			? new[] { methodBody }
			: ((Body)methodCall.Method.GetBodyAndParseIfNeeded()).Expressions;
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
			if (instance.Method.Parameters[parameterIndex].Type is GenericTypeImplementation
				{
					Generic.Name: Base.List
				})
				statements.Add(new StoreVariableStatement(
					new Instance(instance.Method.Parameters[parameterIndex].Type,
						instance.Arguments),
					instance.ReturnType.Members[parameterIndex].Name));
			else
				statements.Add(new StoreVariableStatement(
					new Instance(instance.Arguments[parameterIndex], true),
					instance.ReturnType.Members[parameterIndex].Name));
			if (instance.ReturnType.Members[parameterIndex].Value == null)
				instance.ReturnType.Members[parameterIndex].
					UpdateValue(instance.Arguments[parameterIndex], new Body(instance.Method));
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
		if (new[]
				{
					TryGenerateBodyStatements, TryGenerateBinaryStatements, TryGenerateIfStatements,
					TryGenerateAssignmentStatements, TryGenerateLoopStatements,
					TryGenerateMutableStatements, TryGenerateMemberCallStatement,
					TryGenerateToOperatorStatement, TryGenerateVariableCallStatement,
					TryGenerateMethodCallStatement, TryGenerateValueStatement, TryGenerateListCallStatement
				}.Any(method => method(expression)))
			// ReSharper disable once RedundantJumpStatement
			return;
	}

	private bool TryGenerateListCallStatement(Expression expression)
	{
		if (expression is not ListCall listCall)
			return false;
		GenerateStatementsFromExpression(listCall.Index);
		var indexRegister = registry.PreviousRegister;
		statements.Add(new ListCallStatement(registry.AllocateRegister(), indexRegister,
			listCall.List.ToString()));
		return true;
	}

	private bool TryGenerateToOperatorStatement(Expression expression)
	{
		if (expression is not To toExpression)
			return false;
		if (toExpression.Instance == null)
			return false;
		GenerateStatementsFromExpression(toExpression.Instance);
		statements.Add(new ConversionStatement(registry.PreviousRegister, registry.AllocateRegister(),
			toExpression.ConversionType, GetInstructionForConversionType(toExpression.ConversionType)));
		return true;
	}

	private static Instruction GetInstructionForConversionType(Type conversionType) =>
		conversionType.Name switch
		{
			Base.Text => Instruction.ToText,
			_ => Instruction.ToNumber
		};

	private bool TryGenerateVariableCallStatement(Expression expression)
	{
		if (expression is not (VariableCall or ParameterCall))
			return false;
		statements.Add(new LoadVariableStatement(registry.AllocateRegister(), expression.ToString()));
		return true;
	}

	private bool TryGenerateMemberCallStatement(Expression expression)
	{
		if (expression is not MemberCall memberCall)
			return false;
		if (memberCall.Instance == null)
			statements.Add(
				new LoadVariableStatement(registry.AllocateRegister(), expression.ToString()));
		else if (memberCall.Member.Value != null)
			TryGenerateForEnum(memberCall.Instance.ReturnType, memberCall.Member.Value);
		return true;
	}

	private void TryGenerateForEnum(Type type, Expression value)
	{
		if (type.IsEnum)
			statements.Add(new LoadConstantStatement(registry.AllocateRegister(),
				new Instance(type, value)));
	}

	private bool TryGenerateValueStatement(Expression expression)
	{
		if (expression is not Value valueExpression)
			return false;
		statements.Add(new LoadConstantStatement(registry.AllocateRegister(),
			new Instance(valueExpression.ReturnType, valueExpression.Data)));
		return true;
	}

	private bool TryGenerateMethodCallStatement(Expression expression)
	{
		if (expression is Binary || expression is not MethodCall methodCall)
			return false;
		if (TryGenerateStatementForCollectionManipulation(methodCall))
			return true;
		statements.Add(new InvokeStatement(methodCall, registry.AllocateRegister(), registry));
		return true;
	}

	public bool TryGenerateStatementForCollectionManipulation(MethodCall methodCall)
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
		default:
			return false;
		}
	}

	private void GenerateStatementsForRemoveMethod(MethodCall methodCall)
	{
		if (methodCall.Instance == null)
			return;
		GenerateStatementsFromExpression(methodCall.Arguments[0]);
		if (methodCall.Instance.ReturnType is GenericTypeImplementation { Generic.Name: Base.List })
			statements.Add(new RemoveStatement(methodCall.Instance.ToString(),
				registry.PreviousRegister));
		else
			statements.Add(new RemoveFromTableStatement(registry.PreviousRegister,
				methodCall.Instance.ToString()));
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

	private bool TryGenerateBodyStatements(Expression expression)
	{
		if (expression is not Body body)
			return false;
		GenerateStatements(body.Expressions);
		return true;
	}

	private bool TryGenerateMutableStatements(Expression expression)
	{
		if (expression is MutableDeclaration declaration)
			GenerateForAssignmentOrDeclaration(declaration.Value, declaration.Name);
		else if (expression is MutableAssignment assignment)
			GenerateForAssignmentOrDeclaration(assignment.Value, assignment.Name);
		else
			return false;
		return true;
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
			statements.Add(new StoreFromRegisterStatement(registers[registry.NextRegister - 1], name));
		}
	}

	private bool TryGenerateLoopStatements(Expression expression)
	{
		if (expression is not For forExpression)
			return false;
		GenerateLoopStatements(forExpression);
		return true;
	}

	private bool TryGenerateAssignmentStatements(Expression expression)
	{
		if (expression is not ConstantDeclaration assignmentExpression || expression.IsMutable)
			return false;
		GenerateForAssignmentOrDeclaration(assignmentExpression.Value, assignmentExpression.Name);
		return true;
	}

	private void
		TryGenerateStatementsForAssignmentValue(Value assignmentValue, string variableName) =>
		statements.Add(new StoreVariableStatement(
			new Instance(assignmentValue.ReturnType, assignmentValue.Data), variableName));

	private bool TryGenerateIfStatements(Expression expression)
	{
		if (expression is not If ifExpression)
			return false;
		GenerateIfStatements(ifExpression);
		return true;
	}

	private bool TryGenerateBinaryStatements(Expression expression)
	{
		if (expression is not Binary binary)
			return false;
		GenerateCodeForBinary(binary);
		return true;
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
		GenerateCodeForIfCondition(ifExpression.Condition);
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
		statements.Add(new SetStatement(new Instance(condition.ReturnType, true),
			registry.AllocateRegister()));
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
		GenerateStatementsFromExpression(condition.Arguments[0]);
		return registry.PreviousRegister;
	}

	private Register GenerateLeftSideForIfCondition(Binary condition) =>
		condition.Instance switch
		{
			Binary binaryInstance => GenerateValueBinaryStatements(binaryInstance,
				GetInstructionBasedOnBinaryOperationName(binaryInstance.Method.Name)),
			MethodCall =>
				InvokeAndGetStoredRegisterForConditional(
					condition),
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
			statements.Add(new BinaryStatement(operationInstruction,
				GenerateValueBinaryStatements(binaryOp, operationInstruction),
				registry.AllocateRegister(), registry.AllocateRegister()));
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
			statements.Add(new LoadVariableStatement(left, binary.Instance.ToString()));
		statements.Add(new BinaryStatement(operationInstruction, left, right,
			registry.AllocateRegister()));
	}

	private Register GenerateValueBinaryStatements(MethodCall binary,
		Instruction operationInstruction)
	{
		var (leftRegister, rightRegister, resultRegister) = (registry.AllocateRegister(),
			registry.AllocateRegister(), registry.AllocateRegister());
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