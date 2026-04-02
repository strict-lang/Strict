using Strict.Bytecode;
using Strict.Expressions;
using Strict.Language;
using System.Text;

namespace Strict.Tests;

//ncrunch: no coverage start
internal class Program
{
	public static async Task Main()
	{
		await RunSimpleCalculator();
		await RunAdjustBrightness();
	}

	private static async Task RunSimpleCalculator()
	{
		var binaryFilePath = Path.ChangeExtension(
			Path.Combine(GetExamplesFolder(), "SimpleCalculator.strict"),
			BinaryExecutable.Extension);
		// First, ensure the .strictbinary file exists by compiling from source
		if (!File.Exists(binaryFilePath))
			await new Runner(Path.Combine(AppContext.BaseDirectory, "Examples",
				"SimpleCalculator.strict")).Run();
		// Warm up: one full binary execution to JIT and cache everything (also populates the binary cache)
		await RunBinaryOnce(binaryFilePath);
		Console.WriteLine("Warmup complete. Starting performance measurement...");
		const int Runs = 1000;
		// Measure: 1000 iterations of full Runner.Run() from .strictbinary (cache hits after warmup)
		var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
		var startTicks = DateTime.UtcNow.Ticks;
		for (var run = 0; run < Runs; run++)
			await RunBinaryOnce(binaryFilePath);
		var endTicks = DateTime.UtcNow.Ticks;
		var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
		Console.WriteLine("Total execution time per run (full binary Runner.Run, cached): " +
			TimeSpan.FromTicks(endTicks - startTicks) / Runs);
		Console.WriteLine("Allocated bytes per run (cached): " +
			(allocatedAfter - allocatedBefore) / Runs);
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
		Console.WriteLine("Allocated bytes per run (VM-only): " +
			(hotAllocatedAfter - hotAllocatedBefore) / Runs);
	}

	private static string GetExamplesFolder()
	{
		var path = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "Examples"));
		return Directory.Exists(path)
			? path
			: @"c:\code\GitHub\strict-lang\Strict\Examples";
	}

	private static async Task RunBinaryOnce(string binaryFilePath)
	{
		var saved = Console.Out;
		Console.SetOut(TextWriter.Null);
		try
		{
			await new Runner(binaryFilePath).Run();
		}
		finally
		{
			Console.SetOut(saved);
		}
	}

	private static async Task RunAdjustBrightness()
	{
		Console.WriteLine("Running RunAdjustBrightness");
		var strictBasePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var runner = new Runner(Path.Combine(GetExamplesFolder(), "..", "ImageProcessing",
			"AdjustBrightness.strict"), strictBasePackage);
		var runAllocatedBefore = GC.GetAllocatedBytesForCurrentThread();
		var runStartTicks = DateTime.UtcNow.Ticks;
		const int Runs = 10;
		for (var run = 0; run < Runs; run++)
			await runner.Run();
		var runEndTicks = DateTime.UtcNow.Ticks;
		var runAllocatedAfter = GC.GetAllocatedBytesForCurrentThread();
		Console.WriteLine("RunAdjustBrightness execution time: " +
			TimeSpan.FromTicks(runEndTicks - runStartTicks) / Runs);
		Console.WriteLine("RunAdjustBrightness allocated bytes: " +
			(runAllocatedAfter - runAllocatedBefore) / Runs);
	}
}