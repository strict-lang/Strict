using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class BytecodeGenerator
{
	public BytecodeGenerator(MethodCall methodCall)
	{
		Method = methodCall.Method;
		if (methodCall.Instance != null)
			InstanceArguments = ((MethodCall)methodCall.Instance).Arguments;
		MethodArguments = methodCall.Arguments;
	}

	public IReadOnlyList<Expression> MethodArguments { get; }
	private IReadOnlyList<Expression>? InstanceArguments { get; }
	private Method Method { get; }

	public List<Statement>? Generate()
	{
		var statements = BuildSetStatementsFromParameters();
		var body = (Body)Method.GetBodyAndParseIfNeeded();
		foreach (var expression in body.Expressions)
			if (expression is Binary binary)
				GenerateCodeForBinary(binary, statements);
		return statements;
	}

	private List<Statement>? BuildSetStatementsFromParameters()
	{
		var statements = InstanceArguments?.
			Select(argument => new Statement(Instruction.SetVariable, new Instance(argument))).ToList();
		statements?.AddRange(MethodArguments.Select(argument =>
			new Statement(Instruction.SetVariable, new Instance(argument))));
		return statements;
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
			statements.Add(new LoadStatement(Register.R0));
			statements.Add(new LoadStatement(Register.R1));
			statements.Add(new Statement(Instruction.Add, Register.R0, Register.R1, Register.R2));
		}
	}

	private static void GenerateMultiplyStatements(ICollection<Statement> statements,
		MethodCall binary)
	{
		if (binary.Instance != null)
		{
			statements.Add(new LoadStatement(Register.R0));
			statements.Add(new LoadStatement(Register.R1));
			statements.Add(new Statement(Instruction.Multiply, Register.R0, Register.R1, Register.R2));
		}
	}
}