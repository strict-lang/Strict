using Strict.Expressions;
using Strict.Runtime.Instructions;

namespace Strict.Optimizers.Tests;

public class TestOptimizers
{
	public ValueInstance Num(double value) => new(numberType, value);
	protected readonly Type numberType = TestPackage.Instance.GetType(Type.Number);

	protected List<Instruction> Optimize(InstructionOptimizer optimizer,
		List<Instruction> instructions, int expectedCount)
	{
		var optimizedInstructions = optimizer.Optimize(instructions);
		Assert.That(optimizedInstructions, Has.Count.EqualTo(expectedCount));
		return optimizedInstructions;
	}
}