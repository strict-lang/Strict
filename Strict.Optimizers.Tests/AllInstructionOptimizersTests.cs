using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;

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
		Assert.That(new VirtualMachine(binary).Execute().Returns!.Value.Number, Is.EqualTo(20000));
	}

	[Test]
	public void InlineColorAdjustmentInsideLoopRemovesInvokeFromEntryPoint()
	{
		var binary = CreateColorAdjustmentInliningBinary();
		new AllInstructionOptimizers().Optimize(binary);
		Assert.That(binary.EntryPoint.instructions.OfType<Invoke>().Any(), Is.False);
	}

	internal BinaryExecutable CreateLoopInliningBinary() =>
		GenerateBinary("LoopInlining",
		// @formatter:off
		"has number Number",
		"Run Number",
		"\tmutable temp = number",
		"\tfor 10000",
		"\t\ttemp = AddToNumber(temp, 2)",
		"\ttemp",
		"AddToNumber(temp Number, increase Number) Number",
		"\ttemp + increase");
	// @formatter: on

	internal BinaryExecutable CreateColorAdjustmentInliningBinary()
	{
		var parser = new MethodExpressionParser();
		var package = new Package(TestPackage.Instance, "ColorAdjustmentInliningTests");
		var size = new Type(package, new TypeLines("Size",
			"has Width Number",
			"has Height Number",
			"Length Number",
			"Width * Height")).ParseMembersAndMethods(parser);
		_ = size;
		var color = new Type(package, new TypeLines("Color",
			"has Red Number",
			"has Green Number",
			"has Blue Number")).ParseMembersAndMethods(parser);
		_ = color;
		var colors = new Type(package, new TypeLines("Colors",
			"has elements Color",
			"Length Number",
			"1")).ParseMembersAndMethods(parser);
		_ = colors;
		var colorImage = new Type(package, new TypeLines("ColorImage",
			"has Size",
			"mutable Colors Colors with Length is Size.Length")).ParseMembersAndMethods(parser);
		_ = colorImage;
		var adjustBrightness = new Type(package, new TypeLines("ColorAdjustmentInlining",
			"has brightness Number",
			"Process(mutable image ColorImage) ColorImage",
			"\tfor colorIndex in Range(0, image.Colors.Length)",
			"\t\timage.Colors(colorIndex) = GetBrightnessAdjustedColor(image.Colors(colorIndex))",
			"\timage",
			"GetBrightnessAdjustedColor(current Color) Color",
			"\tColor(current.Red + brightness, current.Green + brightness, current.Blue + brightness)")).ParseMembersAndMethods(parser);
		var processMethod = adjustBrightness.Methods.Single(method => method.Name == "Process");
		return BinaryGenerator.GenerateFromRunMethods(processMethod, [processMethod]);
	}
}