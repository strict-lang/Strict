using Strict.Bytecode.Instructions;
using Strict.Expressions;

namespace Strict.Optimizers.Tests;

public sealed class CompactTypeOptimizerTests : TestOptimizers
{
	[Test]
	public async Task ConvertsConstantColorValueConstructionToColor()
	{
		var repos = new Repositories(new MethodExpressionParser());
		var package = await repos.LoadStrictPackage("Strict/ImageProcessing");
		var colorValueType = package.GetType("ColorValue");
		var colorType = package.GetType("Color");
		var binary = BinaryExecutable.CreateForEntryInstructions(package, [
			new LoadConstantInstruction(Register.R0, Num(0.5)),
			new LoadConstantInstruction(Register.R1, Num(0.5)),
			new LoadConstantInstruction(Register.R2, Num(0.25)),
			new ConstructValueTypeInstruction(Register.R3, colorValueType,
				[Register.R0, Register.R1, Register.R2]),
			new ReturnInstruction(Register.R3)
		]);
		new CompactTypeOptimizer().Optimize(binary);
		var instructions = binary.EntryPoint.instructions;
		Assert.That(((LoadConstantInstruction)instructions[0]).Constant.Number, Is.EqualTo(128));
		Assert.That(((LoadConstantInstruction)instructions[1]).Constant.Number, Is.EqualTo(128));
		Assert.That(((LoadConstantInstruction)instructions[2]).Constant.Number, Is.EqualTo(64));
		Assert.That(instructions[3], Is.InstanceOf<ConstructValueTypeInstruction>());
		Assert.That(((ConstructValueTypeInstruction)instructions[3]).ReturnType, Is.EqualTo(colorType));
	}

	[Test]
	public async Task ConvertsColorValueInvokeToColorConstruction()
	{
		var repos = new Repositories(new MethodExpressionParser());
		var package = await repos.LoadStrictPackage("Strict/ImageProcessing");
		var colorType = package.GetType("Color");
		var binary = BinaryExecutable.CreateForEntryInstructions(package, [
			new LoadConstantInstruction(Register.R0, Num(1)),
			new LoadConstantInstruction(Register.R1, Num(0)),
			new LoadConstantInstruction(Register.R2, Num(0.5)),
			new Invoke(Register.R3, new InvokeMethodInfo(
				"Strict/ImageProcessing/ColorValue", Method.From,
				["red", "green", "blue"], "ColorValue",
				[Register.R0, Register.R1, Register.R2], null)),
			new ReturnInstruction(Register.R3)
		]);
		new CompactTypeOptimizer().Optimize(binary);
		var instructions = binary.EntryPoint.instructions;
		Assert.That(((LoadConstantInstruction)instructions[0]).Constant.Number, Is.EqualTo(255));
		Assert.That(((LoadConstantInstruction)instructions[1]).Constant.Number, Is.EqualTo(0));
		Assert.That(((LoadConstantInstruction)instructions[2]).Constant.Number, Is.EqualTo(128));
		Assert.That(instructions[3], Is.InstanceOf<ConstructValueTypeInstruction>());
		Assert.That(((ConstructValueTypeInstruction)instructions[3]).ReturnType, Is.EqualTo(colorType));
	}

	[Test]
	public async Task DoesNotConvertNonConstantColorValueArguments()
	{
		var repos = new Repositories(new MethodExpressionParser());
		var package = await repos.LoadStrictPackage("Strict/ImageProcessing");
		var colorValueType = package.GetType("ColorValue");
		var binary = BinaryExecutable.CreateForEntryInstructions(package, [
			new LoadVariableToRegister(Register.R0, "dynamicRed"),
			new LoadConstantInstruction(Register.R1, Num(0.5)),
			new LoadConstantInstruction(Register.R2, Num(0.25)),
			new ConstructValueTypeInstruction(Register.R3, colorValueType,
				[Register.R0, Register.R1, Register.R2]),
			new ReturnInstruction(Register.R3)
		]);
		new CompactTypeOptimizer().Optimize(binary);
		var construct = (ConstructValueTypeInstruction)binary.EntryPoint.instructions[3];
		Assert.That(construct.ReturnType, Is.EqualTo(colorValueType),
			"Should not convert when arguments are not all constants");
	}

	[Test]
	public async Task SkipsWhenNoCompactableTypePairsExist()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, Num(5)),
			new ReturnInstruction(Register.R0)
		};
		var binary = BinaryExecutable.CreateForEntryInstructions(
			TestPackage.Instance, instructions);
		new CompactTypeOptimizer().Optimize(binary);
		Assert.That(binary.EntryPoint.instructions, Has.Count.EqualTo(2));
	}

	[Test]
	public async Task ConvertsNumberHolderToByteHolderForSingleMember()
	{
		var repos = new Repositories(new MethodExpressionParser());
		var package = await repos.LoadStrictPackage("Strict/CompactTypeTest");
		var numberHolderType = package.GetType("NumberHolder");
		var byteHolderType = package.GetType("ByteHolder");
		var binary = BinaryExecutable.CreateForEntryInstructions(package, [
			new LoadConstantInstruction(Register.R0, Num(0.5)),
			new ConstructValueTypeInstruction(Register.R1, numberHolderType, [Register.R0]),
			new ReturnInstruction(Register.R1)
		]);
		new CompactTypeOptimizer().Optimize(binary);
		var instructions = binary.EntryPoint.instructions;
		Assert.That(((LoadConstantInstruction)instructions[0]).Constant.Number, Is.EqualTo(128));
		Assert.That(instructions[1], Is.InstanceOf<ConstructValueTypeInstruction>());
		Assert.That(((ConstructValueTypeInstruction)instructions[1]).ReturnType,
			Is.EqualTo(byteHolderType));
	}
}
