using Strict.TestRunner.Tests;

var tests = new TestExecutorTests();
// Warm up
tests.RunAllTestsInPackage();
tests.executor.Statistics.Reset();
var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
var startTicks = DateTime.UtcNow.Ticks;
const int Runs = 100;
for (var count = 0; count < Runs; count++)
	tests.RunAllTestsInPackage();
var endTicks = DateTime.UtcNow.Ticks;
var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
Console.WriteLine("Total execution time per run: " +
	TimeSpan.FromTicks(endTicks - startTicks) / Runs);
Console.WriteLine("Allocated bytes (current thread): " + (allocatedAfter - allocatedBefore));
Console.WriteLine(tests.executor.Statistics);