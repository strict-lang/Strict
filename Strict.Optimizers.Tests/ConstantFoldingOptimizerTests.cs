global using Type = Strict.Language.Type;
using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class ConstantFoldingOptimizerTests : TestOptimizers
{
	[Test]
	public void FoldAdditionOfTwoConstants()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadConstantStatement>());
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(8));
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void FoldSubtractionOfTwoConstants()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(10)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Subtract, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(7));
	}

	[Test]
	public void FoldMultiplicationOfTwoConstants()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(4)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(12));
	}

	[Test]
	public void FoldDivisionOfTwoConstants()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(10)),
			new LoadConstantStatement(Register.R1, Num(2)),
			new Binary(Instruction.Divide, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(5));
	}

	[Test]
	public void FoldModuloOfTwoConstants()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(7)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Modulo, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(1));
	}

	[Test]
	public void FoldTextConcatenation()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, new("Hello")),
			new LoadConstantStatement(Register.R1, new(" World")),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Text, Is.EqualTo("Hello World"));
	}

	[Test]
	public void DoNotFoldWhenOperandsAreNotConstants()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(4));
	}

	[Test]
	public void DoNotFoldConditionalBinaryOperations()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new Return(Register.R0)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(4));
	}

	[Test]
	public void FoldChainedConstants()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(2)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new LoadConstantStatement(Register.R3, Num(4)),
			new Binary(Instruction.Multiply, Register.R2, Register.R3, Register.R4),
			new Return(Register.R4)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(20));
	}

	[Test]
	public void PreserveNonArithmeticStatements()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new Return(Register.R0)
		};
		var optimized = new ConstantFoldingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(3));
	}
}