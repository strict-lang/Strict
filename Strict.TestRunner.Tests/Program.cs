using Strict.TestRunner.Tests;

//ncrunch: no coverage start
var tests = new TestExecutorTests();
// Warm up, will cache a lot of things: first parse, types, bodies, expressions
tests.RunAllTestsInPackage();
Console.WriteLine("Initial warmup run: " + tests.executor.Statistics);
tests.executor.Statistics.Reset();
var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
var startTicks = DateTime.UtcNow.Ticks;
const int Runs = 1000;
for (var count = 0; count < Runs; count++)
	tests.RunAllTestsInPackage();
var endTicks = DateTime.UtcNow.Ticks;
var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
Console.WriteLine("Total execution time per run: " +
	TimeSpan.FromTicks(endTicks - startTicks) / Runs);
Console.WriteLine("Allocated bytes per run: " + (allocatedAfter - allocatedBefore) / Runs);
tests.executor.Statistics.Reset();
tests.RunAllTestsInPackage();
Console.WriteLine("One run: " + tests.executor.Statistics);