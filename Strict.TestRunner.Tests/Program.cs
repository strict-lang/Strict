using Strict.TestRunner.Tests;

var tests = new TestExecutorTests();
// Warm up
tests.RunAllTestsInPackage();
var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
var startTicks = DateTime.UtcNow.Ticks;
tests.RunAllTestsInPackage();
var endTicks = DateTime.UtcNow.Ticks;
var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
Console.WriteLine("Total execution time: " + TimeSpan.FromTicks(endTicks - startTicks));
Console.WriteLine("Allocated bytes (current thread): " + (allocatedAfter - allocatedBefore));
Console.WriteLine("Packages: " + tests.executor.PackagesCount);
Console.WriteLine("Types: " + tests.executor.TypesCount);
Console.WriteLine("Methods: " + tests.executor.MethodsCount);
Console.WriteLine(tests.executor.Statistics);