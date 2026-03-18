using Strict.Bytecode.Instructions;

namespace Strict.Optimizers.Tests;

public sealed class StrengthReducerTests : TestOptimizers
{
	[Test]
	public void MultiplyByOneBecomesLoad()
	{
		var optimizedInstructions = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(1)),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(optimizedInstructions[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimizedInstructions[1], Is.InstanceOf<ReturnInstruction>());
	}

	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new StrengthReducer(), instructions, expectedCount);

	[Test]
	public void MultiplyByOneOnLeftBecomesLoad()
	{
		var optimizedInstructions = Optimize([
			new LoadConstantInstruction(Register.R0, Num(1)),
			new LoadVariableToRegister(Register.R1, "x"),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(optimizedInstructions[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimizedInstructions[1], Is.InstanceOf<ReturnInstruction>());
	}

	[Test]
	public void MultiplyByZeroBecomesLoadZero()
	{
		var optimizedInstructions = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(0)),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(optimizedInstructions[0], Is.InstanceOf<LoadConstantInstruction>());
		Assert.That(((LoadConstantInstruction)optimizedInstructions[0]).Constant.Number,
			Is.EqualTo(0));
	}

	[Test]
	public void AddZeroBecomesLoad()
	{
		var optimizedInstructions = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(optimizedInstructions[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimizedInstructions[1], Is.InstanceOf<ReturnInstruction>());
	}

	[Test]
	public void SubtractZeroBecomesLoad() =>
		Assert.That(Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(0)),
			new BinaryInstruction(InstructionType.Subtract, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0], Is.InstanceOf<LoadVariableToRegister>());

	[Test]
	public void DivideByOneBecomesLoad() =>
		Assert.That(Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(1)),
			new BinaryInstruction(InstructionType.Divide, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0], Is.InstanceOf<LoadVariableToRegister>());

	[Test]
	public void PreserveNonIdentityOperations() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 4);

	[Test]
	public void HandleEmptyInstructionList() => Optimize([], 0);

	[Test]
	public void MultiplyByZeroOnLeftBecomesLoadZero() =>
		Assert.That(((LoadConstantInstruction)Optimize([
			new LoadConstantInstruction(Register.R0, Num(0)),
			new LoadVariableToRegister(Register.R1, "x"),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0]).Constant.Number, Is.EqualTo(0));

	[Test]
	public void AddZeroOnLeftBecomesLoad() =>
		Assert.That(Optimize([
			new LoadConstantInstruction(Register.R0, Num(0)),
			new LoadVariableToRegister(Register.R1, "x"),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 2)[0], Is.InstanceOf<LoadVariableToRegister>());
}