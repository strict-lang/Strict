namespace Strict.Optimizers.Tests;

internal class Program
{
	//ncrunch: no coverage start
	public static void Main()
	{
		var tests = new AllInstructionOptimizersTests();
		var binary = tests.CreateLoopInliningBinary();
		new AllInstructionOptimizers().Optimize(binary);
		var vm = new VirtualMachine(binary);
		vm.Execute();
		for (var iteration = 0; iteration < 100000; iteration++)
			vm.Execute();
	}
}