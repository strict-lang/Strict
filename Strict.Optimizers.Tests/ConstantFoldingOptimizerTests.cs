global using Type = Strict.Language.Type;
using Strict.Bytecode.Instructions;
using Strict.Expressions;

namespace Strict.Optimizers.Tests;

public sealed class ConstantFoldingOptimizerTests : TestOptimizers
{
	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new ConstantFoldingOptimizer(), instructions, expectedCount);

	[Test]
	public void FoldAdditionOfTwoConstants()
	{
		var optimized = Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(optimized[0], Is.InstanceOf<LoadConstantInstruction>());
		Assert.That(((LoadConstantInstruction)optimized[0]).ValueInstance.Number, Is.EqualTo(8));
		Assert.That(optimized[1], Is.InstanceOf<ReturnInstruction>());
	}

	[Test]
	public void FoldSubtractionOfTwoConstants() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(10)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Subtract, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(7));

	[Test]
	public void FoldMultiplicationOfTwoConstants() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(4)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(12));

	[Test]
	public void FoldDivisionOfTwoConstants() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(10)),
			new LoadConstantInstruction(Register.R1, Num(2)),
			new BinaryInstruction(InstructionType.Divide, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(5));

	[Test]
	public void FoldModuloOfTwoConstants() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(7)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Modulo, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(1));

	[Test]
	public void FoldTextConcatenation() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, new("Hello")),
			new LoadConstantInstruction(Register.R1, new(" World")),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0]).ValueInstance.Text, Is.EqualTo("Hello World"));

	[Test]
	public void DoNotFoldWhenOperandsAreNotConstants() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 4);

	[Test]
	public void DoNotFoldConditionalBinaryOperations() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new ReturnInstruction(Register.R0)
		], 4);

	[Test]
	public void FoldChainedConstants() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(2)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new LoadConstantInstruction(Register.R3, Num(4)),
			new BinaryInstruction(InstructionType.Multiply, Register.R2, Register.R3, Register.R4),
			new ReturnInstruction(Register.R4)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(20));

	[Test]
	public void PreserveNonArithmeticInstructions() =>
		Optimize([
			new StoreVariableInstruction(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new ReturnInstruction(Register.R0)
		], 3);

	[Test]
	public void DoNotFoldWithStaleConstantAfterRegisterOverwrite()
	{
		var optimized = new ConstantFoldingOptimizer().Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadVariableToRegister(Register.R1, "celsius"),
			new BinaryInstruction(InstructionType.Multiply, Register.R1, Register.R1, Register.R0),
			new LoadConstantInstruction(Register.R2, Num(32)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R2, Register.R3),
			new ReturnInstruction(Register.R3)
		]);
		var result = new VirtualMachine(TestPackage.Instance).Execute(optimized,
			new Dictionary<string, ValueInstance> { ["celsius"] = Num(100) }).Returns!.Value;
		Assert.That(result.Number, Is.EqualTo(10032));
	}
}