using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

/// <summary>
/// Remove passed test code and provably dead instructions, Tests are executed once; passing
/// expressions become true and are pruned. In Strict, test assertions are comparison + conditional
/// jump blocks at the start of a method. After tests pass, these blocks are dead code and removed.
/// </summary>
public sealed class TestCodeRemoverTests : TestOptimizers
{
	[Test]
	public void RemovePassedTestAssertionPattern()
	{
		// Pattern: load known result, load expected, compare Equal, JumpToIdIfFalse, JumpEnd
		// This is a passed test that always evaluates to true — remove the whole block
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadVariableToRegister(Register.R2, "x"),
			new Return(Register.R2)
		};
		var optimized = new TestCodeRemover().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void RemoveMultiplePassedTestAssertions()
	{
		var statements = new List<Statement>
		{
			// First test: 5 is 5
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			// Second test: 10 is 10
			new LoadConstantStatement(Register.R2, Num(10)),
			new LoadConstantStatement(Register.R3, Num(10)),
			new Binary(Instruction.Equal, Register.R2, Register.R3),
			new JumpToId(Instruction.JumpToIdIfFalse, 1),
			new JumpToId(Instruction.JumpEnd, 1),
			// Actual code
			new LoadVariableToRegister(Register.R4, "result"),
			new Return(Register.R4)
		};
		var optimized = new TestCodeRemover().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
	}

	[Test]
	public void DoNotRemoveTestWithMismatchedConstants()
	{
		// 5 is 3 — these don't match, so this is real conditional logic, not a passed test
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadVariableToRegister(Register.R2, "x"),
			new Return(Register.R2)
		};
		var optimized = new TestCodeRemover().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(7));
	}

	[Test]
	public void DoNotRemoveConditionalBlockWithBody()
	{
		// Test block that has actual code inside - this is real conditional logic
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new LoadConstantStatement(Register.R2, Num(99)),
			new Return(Register.R2),
			new JumpToId(Instruction.JumpEnd, 0),
			new Return(Register.R0)
		};
		var optimized = new TestCodeRemover().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(8));
	}

	[Test]
	public void DoNotRemoveWhenVariablesInvolvedInComparison()
	{
		var statements = new List<Statement>
		{
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new Return(Register.R0)
		};
		var optimized = new TestCodeRemover().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(6));
	}

	[Test]
	public void HandleEmptyStatementList()
	{
		var statements = new List<Statement>();
		var optimized = new TestCodeRemover().Optimize(statements);
		Assert.That(optimized, Is.Empty);
	}
}
