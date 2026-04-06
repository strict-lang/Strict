using BenchmarkDotNet.Running;

namespace BenchmarkSuite1;

internal class Program
{
	private static void Main() => BenchmarkRunner.Run(typeof(Program).Assembly);
}