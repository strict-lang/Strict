using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class BytecodeGenerator
{
	public BytecodeGenerator(MethodCall methodCall)
	{
		InstanceArguments = new Dictionary<string, Expression>();
		Method = methodCall.Method;
		if (methodCall.Instance != null)
			AddInstanceMemberVariables((MethodCall)methodCall.Instance);
		AddMethodParameterVariables(methodCall);
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

	private Dictionary<string, Expression>? InstanceArguments { get; }
	private Method Method { get; }

	public List<Statement> Generate()
	{
		var statements = BuildSetStatementsFromParameters();
		var body = (Body)Method.GetBodyAndParseIfNeeded();
		GenerateCode(body, statements);
		return statements;
	}

	private static void GenerateCode(Body body, ICollection<Statement> statements)
	{
		foreach (var expression in body.Expressions)
			if (expression is Binary binary)
				GenerateCodeForBinary(binary, statements);
	}

	private List<Statement> BuildSetStatementsFromParameters()
	{
		var statements = InstanceArguments?.Select(argument =>
			new StoreStatement(new Instance(argument.Value), argument.Key)).ToList();
		return new List<Statement>(statements!);
	}

	private static void GenerateCodeForBinary(MethodCall binary, ICollection<Statement>? statements)
	{
		if (statements != null)
			switch (binary.Method.Name)
			{
			case BinaryOperator.Plus:
				GenerateAdditionStatements(statements, binary);
				break;
			case BinaryOperator.Multiply:
				GenerateMultiplyStatements(statements, binary);
				break;
			}
	}

	private static void GenerateAdditionStatements(ICollection<Statement> statements,
		MethodCall binary)
	{
		if (binary.Instance != null)
		{
			statements.Add(new LoadVariableStatement(Register.R1, binary.Instance.ToString()));
			statements.Add(new LoadVariableStatement(Register.R0, binary.Arguments[0].ToString()));
			statements.Add(new Statement(Instruction.Add, Register.R1, Register.R0, Register.R2));
		}
	}

	private static void GenerateMultiplyStatements(ICollection<Statement> statements,
		MethodCall binary)
	{
		if (binary.Instance != null)
		{
			statements.Add(new LoadVariableStatement(Register.R1, binary.Instance.ToString()));
			statements.Add(new LoadVariableStatement(Register.R0, binary.Arguments[0].ToString()));
			statements.Add(new Statement(Instruction.Multiply, Register.R0, Register.R1, Register.R2));
		}
	}
}