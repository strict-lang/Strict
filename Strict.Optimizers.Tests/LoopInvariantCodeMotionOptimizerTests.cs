using Strict.Bytecode.Instructions;

namespace Strict.Optimizers.Tests;

public sealed class LoopInvariantCodeMotionOptimizerTests : TestOptimizers
{
	[Test]
	public void LoadIntoRegisterReusedInsideLoopStaysInLoop() =>
		Assert.That(Optimize(new LoopInvariantCodeMotionOptimizer(), [
			new LoadConstantInstruction(Register.R0, Num(3)),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R1, "line"),
			new StoreFromRegisterInstruction(Register.R1, "copy"),
			new LoadVariableToRegister(Register.R1, "index"),
			new StoreFromRegisterInstruction(Register.R1, "position"),
			new LoopEndInstruction(5)
		], 7)[2], Is.InstanceOf<LoadVariableToRegister>());
}
