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
		var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
		var startTicks = DateTime.UtcNow.Ticks;
		const int Runs = 1000;
		for (var iteration = 0; iteration < Runs; iteration++)
			vm.Execute();
		var endTicks = DateTime.UtcNow.Ticks;
		var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
		Console.WriteLine("Total execution time per VirtualMachine.Execute: " +
			TimeSpan.FromTicks(endTicks - startTicks) / Runs);
		Console.WriteLine("Allocated bytes per Execute: " + (allocatedAfter - allocatedBefore) / Runs);
		Assert.That(vm.Execute().Returns!.Value.Number, Is.EqualTo(20000));
	}
}