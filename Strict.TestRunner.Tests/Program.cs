using Strict.TestRunner.Tests;

var tests = new TestExecutorTests();
// Warm up
tests.RunAllTestsInPackage();
var startTicks = DateTime.UtcNow.Ticks;
tests.RunAllTestsInPackage();
var endTicks = DateTime.UtcNow.Ticks;
Console.WriteLine("Total execution time: " + TimeSpan.FromTicks(endTicks - startTicks));
Console.WriteLine("Packages: " + tests.executor.PackagesCount);
Console.WriteLine("Types: " + tests.executor.TypesCount);
Console.WriteLine("Methods: " + tests.executor.MethodsCount);
Console.WriteLine(tests.executor.Statistics);