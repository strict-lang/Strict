using Strict;
using Strict.Bytecode;
using Strict.Expressions;
using Strict.Bytecode.Instructions;

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

	protected ValueInstance ExecuteInstructions(IReadOnlyList<Instruction> instructions,
		IReadOnlyDictionary<string, ValueInstance>? initialVariables = null)
	{
		var binary = BinaryExecutable.CreateForEntryInstructions(TestPackage.Instance, instructions);
		var vm = new VirtualMachine(binary).Execute(initialVariables: initialVariables);
		return vm.Returns!.Value;
	}
}