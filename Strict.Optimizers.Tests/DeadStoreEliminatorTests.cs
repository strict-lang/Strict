using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class DeadStoreEliminatorTests : TestOptimizers
{
	[Test]
	public void RemoveUnusedVariable()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(5), "unused"),
			new StoreVariableStatement(Num(10), "used"),
			new LoadVariableToRegister(Register.R0, "used"),
			new Return(Register.R0)
		};
		var optimized = new DeadStoreEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(3));
		Assert.That(optimized[0], Is.InstanceOf<StoreVariableStatement>());
		Assert.That(((StoreVariableStatement)optimized[0]).Identifier, Is.EqualTo("used"));
	}

	[Test]
	public void KeepVariableThatIsLoaded()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new Return(Register.R0)
		};
		var optimized = new DeadStoreEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(3));
	}

	[Test]
	public void KeepMemberVariables()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(5), "member", isMember: true),
			new LoadConstantStatement(Register.R0, Num(10)),
			new Return(Register.R0)
		};
		var optimized = new DeadStoreEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(3));
	}

	[Test]
	public void RemoveMultipleDeadStores()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(1), "dead1"),
			new StoreVariableStatement(Num(2), "dead2"),
			new StoreVariableStatement(Num(3), "alive"),
			new LoadVariableToRegister(Register.R0, "alive"),
			new Return(Register.R0)
		};
		var optimized = new DeadStoreEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(3));
		Assert.That(((StoreVariableStatement)optimized[0]).Identifier, Is.EqualTo("alive"));
	}

	[Test]
	public void KeepStoreWhenVariableUsedInStoreFromRegister()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(0), "count"),
			new LoadVariableToRegister(Register.R0, "count"),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new StoreFromRegisterStatement(Register.R2, "count"),
			new LoadVariableToRegister(Register.R3, "count"),
			new Return(Register.R3)
		};
		var optimized = new DeadStoreEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(7));
	}

	[Test]
	public void DoNotRemoveEmptyList()
	{
		var statements = new List<Statement>();
		var optimized = new DeadStoreEliminator().Optimize(statements);
		Assert.That(optimized, Is.Empty);
	}
}
