using Strict.Expressions;
using Strict.Expressions.Tests;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers.Tests;

public class TestOptimizers : TestExpressions
{
	public ValueInstance Num(double value) => new(numberType, value);
	protected readonly Type numberType = TestPackage.Instance.GetType(Type.Number);
  protected BinaryExecutable GenerateBinary(string programName, params string[] source)
	{
   var programType = type.Package.FindDirectType(programName) ??
			new Type(type.Package, new TypeLines(programName, source)).ParseMembersAndMethods(this);
		var runMethods = programType.Methods.Where(method => method.Name == Method.Run).ToArray();
		return BinaryGenerator.GenerateFromRunMethods(runMethods[0], runMethods);
	}

	protected List<Instruction> Optimize(InstructionOptimizer optimizer,
		List<Instruction> instructions, int expectedCount)
	{
		var optimizedInstructions = optimizer.Optimize(instructions);
		Assert.That(optimizedInstructions, Has.Count.EqualTo(expectedCount));
		return optimizedInstructions;
	}

	protected ValueInstance ExecuteInstructions(List<Instruction> instructions,
		IReadOnlyDictionary<string, ValueInstance>? initialVariables = null)
	{
		var binary = BinaryExecutable.CreateForEntryInstructions(TestPackage.Instance, instructions);
		var vm = new VirtualMachine(binary).Execute(initialVariables: initialVariables);
		return vm.Returns!.Value;
	}
}