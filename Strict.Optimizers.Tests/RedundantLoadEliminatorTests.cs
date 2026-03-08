using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class RedundantLoadEliminatorTests : TestOptimizers
{
	[Test]
	public void EliminateDuplicateLoadOfSameVariable()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new RedundantLoadEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(3));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		var binaryStatement = (Binary)optimized[1];
		Assert.That(binaryStatement.Registers[0], Is.EqualTo(Register.R0));
		Assert.That(binaryStatement.Registers[1], Is.EqualTo(Register.R0));
	}

	[Test]
	public void DoNotEliminateLoadAfterStoreToSameVariable()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new StoreFromRegisterStatement(Register.R1, "x"),
			new LoadVariableToRegister(Register.R2, "x"),
			new Return(Register.R2)
		};
		var optimized = new RedundantLoadEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(4));
	}

	[Test]
	public void KeepLoadsOfDifferentVariables()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "y"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new RedundantLoadEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(4));
	}

	[Test]
	public void DoNotEliminateLoadAfterLoopBegin()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoopBeginStatement(Register.R0),
			new LoadVariableToRegister(Register.R1, "x"),
			new LoopEndStatement(3),
			new Return(Register.R0)
		};
		var optimized = new RedundantLoadEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(5));
	}

	[Test]
	public void DoNotEliminateWhenNoRedundancy()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new Return(Register.R0)
		};
		var optimized = new RedundantLoadEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(3));
	}
}
