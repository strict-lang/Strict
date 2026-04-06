using System;
using System.IO;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using Microsoft.VSDiagnostics;
using Strict;
using Strict.Bytecode;
using Strict.Expressions;
using Strict.Language;

namespace BenchmarkSuite1;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
[CPUUsageDiagnoser]
public class AdjustBrightnessVmBenchmarks
{
	private BinaryExecutable executable = null!;
	private Package strictBasePackage = null!;
	private string strictFilePath = null!;
	private string binaryFilePath = null!;

	[GlobalSetup]
	public async Task Setup()
	{
		strictBasePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		strictFilePath = Path.Combine(GetExamplesFolder(), "..", "ImageProcessing",
			"AdjustBrightness.strict");
		binaryFilePath = Path.ChangeExtension(strictFilePath, BinaryExecutable.Extension);
		await new Runner(strictFilePath, strictBasePackage).Run();
		executable = new BinaryExecutable(binaryFilePath, strictBasePackage);
	}

	[Benchmark]
	public VirtualMachine ExecuteVmOnly() => new VirtualMachine(executable).Execute();

	private static string GetExamplesFolder()
	{
		var path = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..",
			"..", "..", "Examples"));
		return Directory.Exists(path)
			? path
			: @"c:\code\GitHub\strict-lang\Strict\Examples";
	}
}