using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class ByteCodeGenerator
{
	private readonly Register[] registers = Enum.GetValues<Register>();
	private readonly List<Statement> statements = new();
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
	}

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
			InstanceArguments?.Add(instance.ReturnType.Members[parameterIndex].Name,
				instance.Arguments[parameterIndex]);
	}

	private void AddMethodParameterVariables(MethodCall methodCall)
	{
		for (var parameterIndex = 0; parameterIndex < Method.Parameters.Count; parameterIndex++)
			InstanceArguments?.Add(Method.Parameters[parameterIndex].Name,
				methodCall.Arguments[parameterIndex]);
	}

	public List<Statement> Generate() =>
		GenerateStatements(((Body)Method.GetBodyAndParseIfNeeded()).Expressions);

	private List<Statement> GenerateStatements(IReadOnlyList<Expression> expressions)
	{
		for (var index = 0; index < expressions.Count(); index++)
			if (expressions.Count - 1 == index && expressions[index] is not Assignment and not If
				|| expressions[index] is Return)
				GenerateStatementsFromSeamlessReturn(expressions[index]);
			else
				GenerateStatementsFromExpression(expressions[index]);
		return statements;
	}

	private void GenerateStatementsFromSeamlessReturn(Expression expression)
	{
		if (expression is Return returnExpression)
			GenerateStatementsFromExpression(returnExpression.Value);
		else
			GenerateStatementsFromExpression(expression);
		statements.Add(new ReturnStatement(previousRegister));
	}

	private void GenerateStatementsFromExpression(Expression expression)
	{
		if (expression is Binary binary)
			GenerateCodeForBinary(binary);
		else if (expression is If ifExpression)
			GenerateIfStatements(ifExpression);
		else if (expression is VariableCall variableCall)
			statements.Add(new LoadVariableStatement(AllocateRegister(), variableCall.Name));
		else if (expression is Assignment assignmentExpression)
			statements.Add(new StoreStatement(
				new Instance(assignmentExpression.ReturnType, assignmentExpression.Value),
				assignmentExpression.Name));
	}

	private void GenerateIfStatements(If ifExpression)
	{
		GenerateCodeForIfCondition((Binary)ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
	}

	private void GenerateCodeForThen(If ifExpression) =>
		GenerateStatements(new[] { ifExpression.Then });

	private void GenerateCodeForBinary(MethodCall binary)
	{
		switch (binary.Method.Name)
		{
		case BinaryOperator.Plus:
			GenerateBinaryStatement(binary, Instruction.Add);
			break;
		case BinaryOperator.Multiply:
			GenerateBinaryStatement(binary, Instruction.Multiply);
			break;
		case BinaryOperator.Minus:
			GenerateBinaryStatement(binary, Instruction.Subtract);
			break;
		case BinaryOperator.Divide:
			GenerateBinaryStatement(binary, Instruction.Divide);
			break;
		}
	}

	private void GenerateCodeForIfCondition(MethodCall condition)
	{
		var (leftRegister, rightRegister) = (AllocateRegister(), AllocateRegister());
		if (condition.Instance is Value instanceValue)
			statements.Add(new LoadConstantStatement(leftRegister,
				new Instance(instanceValue.ReturnType, instanceValue.Data)));
		else
			statements.Add(new LoadVariableStatement(leftRegister,
				condition.Instance?.ToString() ?? throw new InvalidOperationException()));
		if (condition.Arguments[0] is Value argumentValue)
			statements.Add(new LoadConstantStatement(rightRegister,
				new Instance(argumentValue.ReturnType, argumentValue.Data)));
		else
			statements.Add(new LoadVariableStatement(rightRegister,
				condition.Arguments[0].ToString()));
		statements.Add(new Statement(Instruction.Equal, leftRegister,
			rightRegister));
		statements.Add(new JumpStatement(Instruction.JumpIfFalse, 4));
	}

	private Register AllocateRegister()
	{
		if (nextRegister == registers.Length)
			nextRegister = 0;
		previousRegister = registers[nextRegister];
		return registers[nextRegister++];
	}

	private void GenerateBinaryStatement(MethodCall binary, Instruction operationInstruction)
	{
		if (binary.Instance != null)
		{
			var (leftRegister, rightRegister) = (AllocateRegister(), AllocateRegister());
			statements.Add(new LoadVariableStatement(leftRegister, binary.Instance.ToString()));
			statements.Add(
				new LoadVariableStatement(rightRegister, binary.Arguments[0].ToString()));
			statements.Add(new Statement(operationInstruction, leftRegister,
				rightRegister, AllocateRegister()));
		}
	}
}