using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class StrengthReducerTests : TestOptimizers
{
	[Test]
	public void MultiplyByOneBecomesLoad()
	{
		var optimizedStatements = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2);
		Assert.That(optimizedStatements[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimizedStatements[1], Is.InstanceOf<Return>());
	}

	private List<Statement> Optimize(List<Statement> statements, int expectedCount) =>
		Optimize(new StrengthReducer(), statements, expectedCount);

	[Test]
	public void MultiplyByOneOnLeftBecomesLoad()
	{
		var optimizedStatements = Optimize([
			new LoadConstantStatement(Register.R0, Num(1)),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2);
		Assert.That(optimizedStatements[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimizedStatements[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void MultiplyByZeroBecomesLoadZero()
	{
		var optimizedStatements = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(0)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2);
		Assert.That(optimizedStatements[0], Is.InstanceOf<LoadConstantStatement>());
		Assert.That(((LoadConstantStatement)optimizedStatements[0]).ValueInstance.Number,
			Is.EqualTo(0));
	}

	[Test]
	public void AddZeroBecomesLoad()
	{
		var optimizedStatements = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(0)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2);
		Assert.That(optimizedStatements[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimizedStatements[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void SubtractZeroBecomesLoad() =>
		Assert.That(Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(0)),
			new Binary(Instruction.Subtract, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2)[0], Is.InstanceOf<LoadVariableToRegister>());

	[Test]
	public void DivideByOneBecomesLoad() =>
		Assert.That(Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Divide, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2)[0], Is.InstanceOf<LoadVariableToRegister>());

	[Test]
	public void PreserveNonIdentityOperations() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 4);

	[Test]
	public void HandleEmptyStatementList() => Optimize([], 0);

	[Test]
	public void MultiplyByZeroOnLeftBecomesLoadZero() =>
		Assert.That(((LoadConstantStatement)Optimize([
			new LoadConstantStatement(Register.R0, Num(0)),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(0));

	[Test]
	public void AddZeroOnLeftBecomesLoad() =>
		Assert.That(Optimize([
			new LoadConstantStatement(Register.R0, Num(0)),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2)[0], Is.InstanceOf<LoadVariableToRegister>());
}