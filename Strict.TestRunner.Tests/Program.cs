using Strict.TestRunner.Tests;

//ncrunch: no coverage start
var tests = new TestInterpreterTests();
// Warm up, will cache a lot of things: first parse, types, bodies, expressions
tests.RunAllTestsInPackage();
Console.WriteLine("Initial warmup run: " + tests.interpreter.Statistics);
tests.interpreter.Statistics.Reset();
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
tests.interpreter.Statistics.Reset();
tests.RunAllTestsInPackage();
Console.WriteLine("One run: " + tests.interpreter.Statistics);
Console.WriteLine("Now running all tests in all .strict files found in this repo!");
allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
startTicks = DateTime.UtcNow.Ticks;
tests.RunAllTestsForAllStrictFilesInThisRepo().GetAwaiter().GetResult();
endTicks = DateTime.UtcNow.Ticks;
allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
Console.WriteLine("Total execution for all tests: " + TimeSpan.FromTicks(endTicks - startTicks));
Console.WriteLine("Allocated bytes for all tests: " + (allocatedAfter - allocatedBefore));
//TODO: the changes are mostly stupid and useless, go through them one by one, still good to have things able to be parallized, but the performance benefit is close to zero and mostly negative (3 times slower running all tests in parallel), too much overhead and not the right way. even now we have extra overhead by locking too many things, we can use some good lock tricks from the past (first return value, then lock if a change is needed, etc.)