using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class UnreachableCodeEliminatorTests : TestOptimizers
{
	[Test]
	public void RemoveStatementsAfterUnconditionalReturn()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new Return(Register.R0),
			new LoadConstantStatement(Register.R1, Num(10)),
			new Return(Register.R1)
		};
		var optimized = new UnreachableCodeEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void KeepAllStatementsWhenNoDeadCodeExists()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new UnreachableCodeEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(4));
	}

	[Test]
	public void DoNotRemoveCodeAfterConditionalJump()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(1)),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new LoadConstantStatement(Register.R2, Num(5)),
			new Return(Register.R2),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R3, Num(10)),
			new Return(Register.R3)
		};
		var optimized = new UnreachableCodeEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(9));
	}

	[Test]
	public void RemoveStatementsAfterReturnInsideConditionalBlock()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new Return(Register.R0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R1, Num(10)),
			new Return(Register.R1)
		};
		var optimized = new UnreachableCodeEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
	}

	[Test]
	public void HandleEmptyStatementList()
	{
		var statements = new List<Statement>();
		var optimized = new UnreachableCodeEliminator().Optimize(statements);
		Assert.That(optimized, Is.Empty);
	}

	[Test]
	public void PreserveCodeInsideLoop()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(0), "sum"),
			new LoadVariableToRegister(Register.R0, "numbers"),
			new LoopBeginStatement(Register.R0),
			new LoadVariableToRegister(Register.R1, "sum"),
			new LoadConstantStatement(Register.R2, Num(1)),
			new Binary(Instruction.Add, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterStatement(Register.R3, "sum"),
			new LoopEndStatement(6),
			new LoadVariableToRegister(Register.R4, "sum"),
			new Return(Register.R4)
		};
		var optimized = new UnreachableCodeEliminator().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(10));
	}
}
