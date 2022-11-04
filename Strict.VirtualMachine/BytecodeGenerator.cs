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

	private List<Statement> GenerateStatements(IEnumerable<Expression> expressions)
	{
		foreach (var expression in expressions)
			GenerateStatementsFromExpression(expression);
		return statements;
	}

	private void GenerateStatementsFromExpression(Expression expression)
	{
		if (expression is Return returnExpression)
		{
			GenerateStatements(new[] { returnExpression.Value });
			statements.Add(new Statement(Instruction.Return, registers[^2]));
		}
		if (expression is Binary binary)
			GenerateCodeForBinary(binary);
		else if (expression is If ifExpression)
			GenerateIfStatements(ifExpression);
	}

	private void GenerateIfStatements(If ifExpression)
	{
		GenerateCodeForIfCondition((Binary)ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
	}

	private void GenerateCodeForThen(If ifExpression) => GenerateStatements(new[] { ifExpression.Then });

	private void GenerateCodeForBinary(MethodCall binary)
	{
		switch (binary.Method.Name)
		{
		case BinaryOperator.Plus:
			GenerateAdditionStatements(binary);
			break;
		case BinaryOperator.Multiply:
			GenerateMultiplyStatements(binary);
			break;
		case BinaryOperator.Minus:
			GenerateSubtractionStatements(binary);
			break;
		case BinaryOperator.Divide:
			GenerateDivisionStatements(binary);
			break;
		}
	}

	private void GenerateCodeForIfCondition(Binary condition)
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
		statements.Add(new JumpStatement(Instruction.JumpIfFalse, 2));
	}

	private Register AllocateRegister() =>
		nextRegister < registers.Length
			? registers[nextRegister++]
			: registers[0];

	private void GenerateAdditionStatements(MethodCall binary) =>
		GenerateBinaryStatements(binary, Instruction.Add);

	private void GenerateSubtractionStatements(MethodCall binary) =>
		GenerateBinaryStatements(binary, Instruction.Subtract);

	private void GenerateDivisionStatements(MethodCall binary) =>
		GenerateBinaryStatements(binary, Instruction.Divide);

	private void GenerateMultiplyStatements(MethodCall binary) =>
		GenerateBinaryStatements(binary, Instruction.Multiply);

	private void GenerateBinaryStatements(MethodCall binary, Instruction operationInstruction)
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