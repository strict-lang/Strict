using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class InstructionOptimizerTests : TestOptimizers
{
	[Test]
	public void ChainsMultipleOptimizers()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(99), "unused"),
			new LoadConstantStatement(Register.R0, Num(2)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(optimized[0], Is.InstanceOf<LoadConstantStatement>());
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(5));
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void OptimizeConstantAssignmentExpression()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(5), "number"),
			new StoreVariableStatement(Num(5), "five"),
			new LoadVariableToRegister(Register.R0, "five"),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new StoreFromRegisterStatement(Register.R2, "something"),
			new LoadVariableToRegister(Register.R3, "something"),
			new LoadConstantStatement(Register.R4, Num(10)),
			new Binary(Instruction.Add, Register.R3, Register.R4, Register.R5),
			new Return(Register.R5)
		};
		var originalCount = statements.Count;
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized.Count, Is.LessThan(originalCount));
	}

	[Test]
	public void PreserveLoopStatements()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(10), "number"),
			new StoreVariableStatement(Num(1), "result"),
			new StoreVariableStatement(Num(2), "multiplier"),
			new LoadVariableToRegister(Register.R0, "number"),
			new LoopBeginStatement(Register.R0),
			new LoadVariableToRegister(Register.R1, "result"),
			new LoadVariableToRegister(Register.R2, "multiplier"),
			new Binary(Instruction.Multiply, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterStatement(Register.R3, "result"),
			new LoopEndStatement(7),
			new LoadVariableToRegister(Register.R4, "result"),
			new Return(Register.R4)
		};
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized.Any(s => s is LoopBeginStatement));
		Assert.That(optimized.Any(s => s is LoopEndStatement));
	}

	[Test]
	public void OptimizeWithRedundantLoads()
	{
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(4));
		var binaryStatement = (Binary)optimized[2];
		Assert.That(binaryStatement.Registers[0], Is.EqualTo(Register.R0));
		Assert.That(binaryStatement.Registers[1], Is.EqualTo(Register.R0));
	}

	[Test]
	public void OptimizedStatementsExecuteCorrectly()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(10)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var optimized = new InstructionOptimizer().Optimize(statements);
		var vm = new BytecodeInterpreter(TestPackage.Instance);
		var result = vm.Execute(optimized).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(15));
	}

	[Test]
	public void OptimizedMultiplicationExecutesCorrectly()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(4)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new LoadConstantStatement(Register.R3, Num(2)),
			new Binary(Instruction.Add, Register.R2, Register.R3, Register.R4),
			new Return(Register.R4)
		};
		var optimized = new InstructionOptimizer().Optimize(statements);
		var vm = new BytecodeInterpreter(TestPackage.Instance);
		var result = vm.Execute(optimized).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(14));
	}

	[Test]
	public void EmptyListRemainsEmpty()
	{
		var statements = new List<Statement>();
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized, Is.Empty);
	}

	[Test]
	public void PipelineRemovesPassedTestsThenFoldsConstants()
	{
		// Test assertion (5 is 5) followed by constant arithmetic (2 + 3)
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R2, Num(2)),
			new LoadConstantStatement(Register.R3, Num(3)),
			new Binary(Instruction.Add, Register.R2, Register.R3, Register.R4),
			new Return(Register.R4)
		};
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(5));
	}

	[Test]
	public void PipelineReducesStrengthAndEliminatesDeadStores()
	{
		// unused store + x * 1 (identity)
		var statements = new List<Statement>
		{
			new StoreVariableStatement(Num(42), "unused"),
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		};
		var originalCount = statements.Count;
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized.Count, Is.LessThan(originalCount));
		Assert.That(optimized.Any(s => s is LoadVariableToRegister));
		Assert.That(optimized[^1], Is.InstanceOf<Return>());
	}

	[Test]
	public void PipelineRemovesUnreachableCodeAfterFolding()
	{
		// After constant folding: LoadConst 8, Return, then some dead code
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2),
			new LoadConstantStatement(Register.R3, Num(999)),
			new Return(Register.R3)
		};
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(2));
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(8));
	}

	[Test]
	public void PipelineHandlesComplexMethodWithTestsAndIdentity()
	{
		// Full Strict method pattern: test assertion + identity operation + return
		var statements = new List<Statement>
		{
			// Test: 10 is 10 (passed test, should be removed)
			new LoadConstantStatement(Register.R0, Num(10)),
			new LoadConstantStatement(Register.R1, Num(10)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			// Real code: x + 0 (identity, should reduce to just x)
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R2, "x"),
			new LoadConstantStatement(Register.R3, Num(0)),
			new Binary(Instruction.Add, Register.R2, Register.R3, Register.R4),
			new Return(Register.R4)
		};
		var originalCount = statements.Count;
		var optimized = new InstructionOptimizer().Optimize(statements);
		Assert.That(optimized.Count, Is.LessThan(originalCount));
		Assert.That(optimized[^1], Is.InstanceOf<Return>());
	}
}
