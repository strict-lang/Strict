using NUnit.Framework;
using Strict.Language;
using Strict.Language.Expressions.Tests;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine.Tests;

public sealed class VirtualMachineTests : TestExpressions
{
	private static readonly Type NumberType = new TestPackage().FindType(Base.Number)!;
	private static readonly Type TextType = new TestPackage().FindType(Base.Text)!;

	[SetUp]
	public void Setup() => vm = new VirtualMachine();

	private VirtualMachine vm = null!;

	[TestCase(Instruction.Add, 15, 5, 10)]
	[TestCase(Instruction.Subtract, 5, 8, 3)]
	[TestCase(Instruction.Multiply, 4, 2, 2)]
	[TestCase(Instruction.Divide, 3, 7.5, 2.5)]
	[TestCase(Instruction.Add, "105", "5", 10)]
	[TestCase(Instruction.Add, "510", 5, "10")]
	[TestCase(Instruction.Add, "510", "5", "10")]
	public void Execute(Instruction operation, object expected, params object[] inputs) =>
		Assert.That(vm.Execute(BuildStatements(inputs, operation))[Register.R1].Value,
			Is.EqualTo(expected));

	private static Statement[]
		BuildStatements(IReadOnlyList<object> inputs, Instruction operation) =>
		new Statement[]
		{
			new(Instruction.Set, new Instance(inputs[0] is int
				? NumberType
				: TextType, inputs[0]), Register.R0),
			new(Instruction.Set, new Instance(inputs[1] is int
				? NumberType
				: TextType, inputs[1]), Register.R1),
			new(operation, Register.R0, Register.R1)
		};

	[Test]
	public void LoadVariable() =>
		Assert.That(
			vm.Execute(new Statement[]
			{
				new(Instruction.SetVariable, new Instance(NumberType, 5)),
				new LoadStatement(Register.R0)
			})[Register.R0].Value, Is.EqualTo(5));

	[Test]
	public void SetAndAdd() =>
		Assert.That(vm.Execute(new Statement[]
		{
			new(Instruction.SetVariable, new Instance(NumberType, 10)),
			new(Instruction.SetVariable, new Instance(NumberType, 5)), new LoadStatement(Register.R0),
			new LoadStatement(Register.R1),
			new(Instruction.Add, Register.R0, Register.R1, Register.R2)
		})[Register.R2].Value, Is.EqualTo(15));

	[Test]
	public void AddFiveTimes() =>
		Assert.That(vm.Execute(new Statement[]
		{
			new(Instruction.Set, new Instance(NumberType, 5), Register.R0),
			new(Instruction.Set, new Instance(NumberType, 1), Register.R1),
			new(Instruction.Set, new Instance(NumberType, 0), Register.R2),
			new(Instruction.Add, Register.R0, Register.R2, Register.R2), // R2 = R0 + R2
			new(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
			new(Instruction.JumpIfNotZero, new Instance(NumberType, -3), Register.R0)
		})[Register.R2].Value, Is.EqualTo(0 + 5 + 4 + 3 + 2 + 1));

	[Test]
	public void ConditionalJump() =>
		Assert.That(
			vm.Execute(new Statement[]
			{
				new(Instruction.Set, new Instance(NumberType, 5), Register.R0),
				new(Instruction.Set, new Instance(NumberType, 1), Register.R1),
				new(Instruction.Set, new Instance(NumberType, 10), Register.R2),
				new(Instruction.LessThan, Register.R2, Register.R0),
				new(Instruction.JumpIfTrue, new Instance(NumberType, 2)),
				new(Instruction.Add, Register.R2, Register.R0, Register.R0)
			})[Register.R0].Value, Is.EqualTo(15));

	// if r0 conditional r1 then r0 = r0 + r1 else r0 = r0 - r1
	[TestCase(Instruction.GreaterThan, new[] { 1, 2 }, 2 - 1)]
	[TestCase(Instruction.LessThan, new[] { 1, 2 }, 1 + 2)]
	[TestCase(Instruction.Equal, new[] { 5, 5 }, 5 + 5)]
	[TestCase(Instruction.NotEqual, new[] { 5, 5 }, 5 - 5)]
	public void ConditionalJumpIfAndElse(Instruction conditional, int[] registers, int expected) =>
		Assert.That(vm.Execute(new Statement[]
		{
			new(Instruction.Set, new Instance(NumberType, registers[0]), Register.R0),
			new(Instruction.Set, new Instance(NumberType, registers[1]), Register.R1),
			new(conditional, Register.R0, Register.R1),
			new(Instruction.JumpIfTrue, new Instance(NumberType, 2)),
			new(Instruction.Subtract, Register.R1, Register.R0, Register.R0),
			new(Instruction.JumpIfFalse,
				new Instance(NumberType, 2)),
			new(Instruction.Add, Register.R0, Register.R1, Register.R0)
		})[Register.R0].Value, Is.EqualTo(expected));

	[TestCase(Instruction.Add)]
	[TestCase(Instruction.GreaterThan)]
	public void OperandsRequired(Instruction instruction) =>
		Assert.That(() => vm.Execute(new Statement[] { new(instruction, Register.R0) }),
			Throws.InstanceOf<VirtualMachine.OperandsRequired>());
}