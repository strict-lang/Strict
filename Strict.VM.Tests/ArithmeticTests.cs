using NUnit.Framework;
using static Strict.VM.VirtualMachine;

namespace Strict.VM.Tests;

public sealed class ArithmeticTests
{
	[SetUp]
	public void Setup() => machine = new VirtualMachine();

	private VirtualMachine machine = null!;

	[TestCase(new[] { 5.0, 10 }, Instruction.Add, 15)]
	[TestCase(new[] { 8.0, 3 }, Instruction.Subtract, 5)]
	[TestCase(new[] { 2.0, 2 }, Instruction.Multiply, 4)]
	[TestCase(new[] { 7.5, 2.5 }, Instruction.Divide, 3)]
	public void Execute(double[] inputs, Instruction operation, float expected) =>
		Assert.That(machine.Execute(BuildStatements(inputs, operation)), Is.EqualTo(expected));

	private static Statement[] BuildStatements(IReadOnlyList<double> inputs, Instruction operation) =>
		new Statement[]
		{
			new(Instruction.Push, inputs[0]),
			new(Instruction.Push, inputs[1]),
			new(operation)
		};

	[Test]
	public void AddFiveTimes() =>
		Assert.That(
			machine.Execute(new Statement[]
			{
				//new(Instruction.Set, Register.A, 5),
				new(Instruction.Push, 1),
				new(Instruction.Push, 2), // jumps here
				new(Instruction.Add),
				new(Instruction.Push, 1),
				//new(Instruction.Subtract, Register.A),
				new(Instruction.JumpIfNotZero, 5)
			}), Is.EqualTo(1 + 2 + 2 + 2 + 2 + 2));












	//TODO: inputs and outputs should not just be doubles, but actual Value or strict expressions
	/*
	public void SimpleArithmeticAddition()
	{
		var instructions = new Statement[]
		{
			new(Instruction.Push, 5),
			new(Instruction.Push, 10),
			new(Instruction.Add)
		};
		machine.Run(instructions);
		Assert.That(machine.register.Stack.Pop(), Is.EqualTo(20));
	}

	[Test]
	public void StackElementDoesNotExistWhenInstructionIsCalled()
	{
		var instructions = new Statement[] { new(Instruction.Add), new(Instruction.Quit) };
		Assert.That(() => machine.Run(instructions),
			Throws.InstanceOf<StackElementDoesNotExistWhenInstructionIsCalled>());
	}*/
}