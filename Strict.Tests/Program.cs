using Strict;
using Strict.Bytecode.Serialization;
using Strict.Tests;

//ncrunch: no coverage start
var binaryFilePath = Path.ChangeExtension(
	Path.Combine(AppContext.BaseDirectory, "Examples", "SimpleCalculator.strict"),
	BytecodeSerializer.Extension);
// First, ensure the .strictbinary file exists by compiling from source
if (!File.Exists(binaryFilePath))
	RunSilently(() => new Runner(
		Path.Combine(AppContext.BaseDirectory, "Examples", "SimpleCalculator.strict")).Run().Dispose());
// Warm up: one full binary execution to JIT and cache everything (also populates the binary cache)
RunBinaryOnce(binaryFilePath);
Console.WriteLine("Warmup complete. Starting performance measurement...");
const int Runs = 1000;
// Measure: 1000 iterations of full Runner.Run() from .strictbinary (cache hits after warmup)
var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
var startTicks = DateTime.UtcNow.Ticks;
for (var run = 0; run < Runs; run++)
	RunBinaryOnce(binaryFilePath);
var endTicks = DateTime.UtcNow.Ticks;
var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
Console.WriteLine("Total execution time per run (full binary Runner.Run, cached): " +
	TimeSpan.FromTicks(endTicks - startTicks) / Runs);
Console.WriteLine("Allocated bytes per run (cached): " + (allocatedAfter - allocatedBefore) / Runs);
// Now measure only the hot VM execution loop (pre-loaded bytecode, no file I/O)
var hotPathBenchmark = new BinaryExecutionPerformanceTests();
await hotPathBenchmark.ExecuteBinary();
Console.WriteLine("Warmup (VM-only) complete. Measuring VM-only hot path...");
var hotAllocatedBefore = GC.GetAllocatedBytesForCurrentThread();
var hotStartTicks = DateTime.UtcNow.Ticks;
await hotPathBenchmark.ExecuteBinary1000Times();
var hotEndTicks = DateTime.UtcNow.Ticks;
var hotAllocatedAfter = GC.GetAllocatedBytesForCurrentThread();
Console.WriteLine("Total execution time per run (VM-only, pre-loaded bytecode): " +
	TimeSpan.FromTicks(hotEndTicks - hotStartTicks) / Runs);
Console.WriteLine("Allocated bytes per run (VM-only): " + (hotAllocatedAfter - hotAllocatedBefore) / Runs);

static void RunBinaryOnce(string binaryFilePath) =>
	RunSilently(() => new Runner(binaryFilePath).Run().Dispose());

static void RunSilently(Action action)
{
	var saved = Console.Out;
	Console.SetOut(TextWriter.Null);
	try
	{
		action();
	}
	finally
	{
		Console.SetOut(saved);
	}
}