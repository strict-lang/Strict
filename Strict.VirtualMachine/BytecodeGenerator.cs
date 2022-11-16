using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class ByteCodeGenerator
{
	public ByteCodeGenerator(MethodCall methodCall)
	{
		InstanceArguments = new Dictionary<string, Expression>();
		Method = methodCall.Method;
		if (methodCall.Instance != null)
			AddInstanceMemberVariables((MethodCall)methodCall.Instance);
		AddMethodParameterVariables(methodCall);
		StoreAndLoadVariables();
	}

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

	private readonly Register[] registers = Enum.GetValues<Register>();
	private int nextRegister;
	private Dictionary<string, Expression>? InstanceArguments { get; }
	private Method Method { get; }
	private readonly List<Statement> statements = new();

	public List<Statement> Generate() =>
		GenerateStatements(((Body)Method.GetBodyAndParseIfNeeded()).Expressions);

	private List<Statement> GenerateStatements(IReadOnlyList<Expression> expressions)
	{
		for (var index = 0; index < expressions.Count(); index++)
		{
			if (expressions.Count - 1 == index && expressions[index] is VariableCall variableCall)
				GenerateStatementsFromSeamlessReturn(variableCall);
			GenerateStatementsFromExpression(expressions[index]);
		}
		return statements;
	}

	private void GenerateStatementsFromSeamlessReturn(VariableCall variableCall)
	{
		var registerToReturn = AllocateRegister();
		statements.Add(new LoadVariableStatement(registerToReturn, variableCall.Name));
		statements.Add(new ReturnStatement(registerToReturn));
	}

	private void GenerateStatementsFromExpression(Expression expression)
	{
		if (expression is Return returnExpression)
		{
			GenerateStatements(new[] { returnExpression.Value });
			statements.Add(new ReturnStatement(registers[^2]));
		}
		if (expression is Binary binary)
			GenerateCodeForBinary(binary);
		else if (expression is If ifExpression)
			GenerateIfStatements(ifExpression);
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
		if (condition.Instance is Value instanceValue)
			statements.Add(new LoadConstantStatement(AllocateRegister(),
				new Instance(instanceValue.ReturnType, instanceValue.Data)));
		else
			statements.Add(new LoadVariableStatement(AllocateRegister(),
				condition.Instance?.ToString() ?? throw new InvalidOperationException()));
		if (condition.Arguments[0] is Value argumentValue)
			statements.Add(new LoadConstantStatement(AllocateRegister(),
				new Instance(argumentValue.ReturnType, argumentValue.Data)));
		else
			statements.Add(new LoadVariableStatement(AllocateRegister(),
				condition.Arguments[0].ToString()));
		statements.Add(new Statement(Instruction.Equal, registers[nextRegister - 2],
			registers[nextRegister - 1]));
		statements.Add(new JumpStatement(Instruction.JumpIfFalse, 4));
	}

	private Register AllocateRegister() =>
		nextRegister < registers.Length
			? registers[nextRegister++]
			: registers[0];

	private void GenerateBinaryStatement(MethodCall binary, Instruction operationInstruction)
	{
		if (binary.Instance != null)
		{
			statements.Add(new LoadVariableStatement(AllocateRegister(), binary.Instance.ToString()));
			statements.Add(
				new LoadVariableStatement(AllocateRegister(), binary.Arguments[0].ToString()));
			statements.Add(new Statement(operationInstruction, registers[nextRegister - 2],
				registers[nextRegister - 1], registers[nextRegister - 2]));
			nextRegister = 0;
		}
	}
}