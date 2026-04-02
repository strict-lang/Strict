using Strict.Bytecode.Instructions;

namespace Strict.Optimizers.Tests;

public sealed class AllInstructionOptimizersTests : TestOptimizers
{
	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new AllInstructionOptimizers(), instructions, expectedCount);

	[Test]
	public void ChainsMultipleOptimizers()
	{
		var optimized = Optimize([
			new StoreVariableInstruction(Num(99), "unused"),
			new LoadConstantInstruction(Register.R0, Num(2)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(((LoadConstantInstruction)optimized[0]).Constant.Number, Is.EqualTo(5));
		Assert.That(optimized[1], Is.InstanceOf<ReturnInstruction>());
	}

	[Test]
	public void OptimizeConstantAssignmentExpression() =>
		Optimize([
			new StoreVariableInstruction(Num(5), "number"),
			new StoreVariableInstruction(Num(5), "five"),
			new LoadVariableToRegister(Register.R0, "five"),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new StoreFromRegisterInstruction(Register.R2, "something"),
			new LoadVariableToRegister(Register.R3, "something"),
			new LoadConstantInstruction(Register.R4, Num(10)),
			new BinaryInstruction(InstructionType.Add, Register.R3, Register.R4, Register.R5),
			new ReturnInstruction(Register.R5)
		], 9);

	[Test]
	public void PreserveLoopInstructions()
	{
		var optimized = Optimize([
			new StoreVariableInstruction(Num(10), "number"),
			new StoreVariableInstruction(Num(1), "result"),
			new StoreVariableInstruction(Num(2), "multiplier"),
			new LoadVariableToRegister(Register.R0, "number"),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R1, "result"),
			new LoadVariableToRegister(Register.R2, "multiplier"),
			new BinaryInstruction(InstructionType.Multiply, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterInstruction(Register.R3, "result"),
			new LoopEndInstruction(7),
			new LoadVariableToRegister(Register.R4, "result"),
			new ReturnInstruction(Register.R4)
		], 12);
		Assert.That(optimized.Any(s => s is LoopBeginInstruction));
		Assert.That(optimized.Any(s => s is LoopEndInstruction));
	}

	[Test]
	public void OptimizeWithRedundantLoads() =>
		Assert.That(((BinaryInstruction)Optimize([
			new StoreVariableInstruction(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 4)[2]).Registers, Is.EqualTo(new[] { Register.R0, Register.R0, Register.R2 }));

	[Test]
	public void OptimizedInstructionsExecuteCorrectly() =>
		Assert.That(ExecuteInstructions(Optimize([
			new LoadConstantInstruction(Register.R0, Num(10)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)).Number, Is.EqualTo(15));

	[Test]
	public void OptimizedMultiplicationExecutesCorrectly() =>
		Assert.That(ExecuteInstructions(Optimize([
			new LoadConstantInstruction(Register.R0, Num(4)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new LoadConstantInstruction(Register.R3, Num(2)),
			new BinaryInstruction(InstructionType.Add, Register.R2, Register.R3, Register.R4),
			new ReturnInstruction(Register.R4)
		], 2)).Number, Is.EqualTo(14));

	[Test]
	public void EmptyListRemainsEmpty() => Optimize([], 0);

	[Test]
	public void PipelineRemovesPassedTestsThenFoldsConstants() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(0, InstructionType.JumpToIdIfFalse),
			new JumpToId(0, InstructionType.JumpEnd),
			new LoadConstantInstruction(Register.R2, Num(2)),
			new LoadConstantInstruction(Register.R3, Num(3)),
			new BinaryInstruction(InstructionType.Add, Register.R2, Register.R3, Register.R4),
			new ReturnInstruction(Register.R4)
		], 2)[0]).Constant.Number, Is.EqualTo(5));

	[Test]
	public void PipelineReducesStrengthAndEliminatesDeadStores()
	{
		var optimized = Optimize([
			new StoreVariableInstruction(Num(42), "unused"),
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(1)),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(optimized.Any(s => s is LoadVariableToRegister));
		Assert.That(optimized[^1], Is.InstanceOf<ReturnInstruction>());
	}

	[Test]
	public void PipelineRemovesUnreachableCodeAfterFolding() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2),
			new LoadConstantInstruction(Register.R3, Num(999)),
			new ReturnInstruction(Register.R3)
		], 2)[0]).Constant.Number, Is.EqualTo(8));

	[Test]
	public void PipelineHandlesComplexMethodWithTestsAndIdentity() =>
		Assert.That(Optimize([
			new LoadConstantInstruction(Register.R0, Num(10)),
			new LoadConstantInstruction(Register.R1, Num(10)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(0, InstructionType.JumpToIdIfFalse),
			new JumpToId(0, InstructionType.JumpEnd),
			new StoreVariableInstruction(Num(5), "x"),
			new LoadVariableToRegister(Register.R2, "x"),
			new LoadConstantInstruction(Register.R3, Num(0)),
			new BinaryInstruction(InstructionType.Add, Register.R2, Register.R3, Register.R4),
			new ReturnInstruction(Register.R4)
		], 3)[^1], Is.InstanceOf<ReturnInstruction>());

	[Test]
	public void InlineOneLineMethodInsideLoop()
	{
		var binary = CreateLoopInliningBinary();
		new AllInstructionOptimizers().Optimize(binary);
		Assert.That(binary.EntryPoint.instructions.OfType<Invoke>().Any(), Is.False);
		Assert.That(new VirtualMachine(binary).Execute().Returns!.Value.Number, Is.EqualTo(2000));
	}

	internal BinaryExecutable CreateLoopInliningBinary() =>
		GenerateBinary("LoopInlining",
		// @formatter:off
		"has number Number",
		"Run Number",
		"\tmutable temp = number",
		"\tfor 1000",
		"\t\ttemp = AddToNumber(temp, 2)",
		"\ttemp",
		"AddToNumber(temp Number, increase Number) Number",
		"\ttemp + increase");
	// @formatter: on
}