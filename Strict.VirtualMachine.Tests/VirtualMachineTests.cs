using NUnit.Framework;

namespace Strict.VirtualMachine.Tests;

public sealed class VirtualMachineTests
{
	[SetUp]
	public void Setup() => machine = new VirtualMachine();

	private VirtualMachine machine = null!;

	[TestCase(new[] { 5.0, 10 }, Instruction.Add, 15)]
	[TestCase(new[] { 8.0, 3 }, Instruction.Subtract, 5)]
	[TestCase(new[] { 2.0, 2 }, Instruction.Multiply, 4)]
	[TestCase(new[] { 7.5, 2.5 }, Instruction.Divide, 3)]
	public void Execute(double[] inputs, Instruction operation, float expected) =>
		Assert.That(machine.Execute(BuildStatements(inputs, operation))[Register.R1],
			Is.EqualTo(expected));

	private static Statement[]
		BuildStatements(IReadOnlyList<double> inputs, Instruction operation) =>
		new Statement[]
		{
			new(Instruction.Set, inputs[0], Register.R0),
			new(Instruction.Set, inputs[1], Register.R1),
			new(operation, Register.R0, Register.R1)
		};

	[Test]
	public void AddFiveTimes() =>
		Assert.That(machine.Execute(new Statement[]
		{
			new(Instruction.Set, 5, Register.R0),
			new(Instruction.Set, 1, Register.R1),
			new(Instruction.Set, Register.R2), //initialized with 0
			new(Instruction.Add, Register.R0, Register.R2, Register.R2), // R2 = R0 + R2
			new(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
			new(Instruction.JumpIfNotZero, -3, Register.R0)
		})[Register.R2], Is.EqualTo(0 + 5 + 4 + 3 + 2 + 1));

	[Test]
	public void ConditionalJump() =>
		Assert.That(
			machine.Execute(new Statement[]
			{
				new(Instruction.Set, 5, Register.R0),
				new(Instruction.Set, 1, Register.R1),
				new(Instruction.Set, 10, Register.R2),
				new(Instruction.LessThan, Register.R2, Register.R0), new(Instruction.JumpIfTrue, 2),
				new(Instruction.Add, Register.R2, Register.R0, Register.R0)
			})[Register.R0], Is.EqualTo(15));

	// if r0 conditional r1 then r0 = r0 + r1 else r0 = r0 - r1
	[TestCase(Instruction.GreaterThan, new[] { 1, 2 }, 2 - 1)]
	[TestCase(Instruction.LessThan, new[] { 1, 2 }, 1 + 2)]
	[TestCase(Instruction.Equal, new[] { 5, 5 }, 5 + 5)]
	[TestCase(Instruction.NotEqual, new[] { 5, 5 }, 5 - 5)]
	public void ConditionalJumpIfAndElse(Instruction conditional, int[] registers, int expected) =>
		Assert.That(machine.Execute(new Statement[]
		{
			new(Instruction.Set, registers[0], Register.R0),
			new(Instruction.Set, registers[1], Register.R1),
			new(conditional, Register.R0, Register.R1),
			new(Instruction.JumpIfTrue, 2),
			new(Instruction.Subtract, Register.R1, Register.R0, Register.R0), //else
			new(Instruction.JumpIfFalse, 2), // if above condition was false, skip the next one
			new(Instruction.Add, Register.R0, Register.R1, Register.R0) //if
		})[Register.R0], Is.EqualTo(expected));

	[TestCase(Instruction.Add)]
	[TestCase(Instruction.GreaterThan)]
	public void OperandsRequired(Instruction instruction) =>
		Assert.That(() => machine.Execute(new Statement[] { new(instruction, Register.R0) }),
			Throws.InstanceOf<VirtualMachine.OperandsRequired>());
}