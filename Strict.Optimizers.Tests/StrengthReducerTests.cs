using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class StrengthReducerTests
{
	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);
	private static ValueInstance Number(double value) => new(NumberType, value);

	[Test]
	public void MultiplyByOneBecomesLoad()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Number(1)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void MultiplyByOneOnLeftBecomesLoad()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Number(1)),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void MultiplyByZeroBecomesLoadZero()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Number(0)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadConstantStatement>());
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(0));
	}

	[Test]
	public void AddZeroBecomesLoad()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Number(0)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void SubtractZeroBecomesLoad()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Number(0)),
			new Binary(Instruction.Subtract, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
	}

	[Test]
	public void DivideByOneBecomesLoad()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Number(1)),
			new Binary(Instruction.Divide, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
	}

	[Test]
	public void PreserveNonIdentityOperations()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Number(5)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(4));
	}

	[Test]
	public void HandleEmptyStatementList()
	{
		var statements = new List<Statement>();
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Is.Empty);
	}

	[Test]
	public void MultiplyByZeroOnLeftBecomesLoadZero()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Number(0)),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(0));
	}

	[Test]
	public void AddZeroOnLeftBecomesLoad()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Number(0)),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new StrengthReducer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
	}
}
