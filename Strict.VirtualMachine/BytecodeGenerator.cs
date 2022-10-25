using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class BytecodeGenerator
{
	public static List<Statement> Generate(Method method)
	{
		var statements = method.Parameters.Select(parameter =>
				new Statement(Instruction.SetVariable,
					new Instance(parameter.Type, 0, parameter.Name))).
			ToList();
		var body = method.GetBodyAndParseIfNeeded();
		if (body is Binary binary)
			switch (binary.Method.Name)
			{
			case BinaryOperator.Plus:
				GenerateAdditionStatements(statements, binary);
				break;
			}
		return statements;
	}

	private static void GenerateAdditionStatements(ICollection<Statement> statements,
		MethodCall binary)
	{
		if (binary.Instance != null)
		{
			statements.Add(new LoadStatement(binary.Instance.ToString(), Register.R0));
			statements.Add(new LoadStatement(binary.Arguments[0].ToString(), Register.R1));
			statements.Add(new Statement(Instruction.Add, Register.R0, Register.R1, Register.R2));
		}
	}
}