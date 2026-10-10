using System.Text;

namespace Strict.Expressions.Tests;

public sealed class NativeProcessRunnerTests
{
	[Test]
	[Category("Slow")]
	public void OutputIsCompleteWhileThreadPoolIsBusy()
	{
		ThreadPool.GetMinThreads(out var workers, out _);
		var busyWorkers = Enumerable.Range(0, workers * 4).Select(_ => Task.Run(() => Thread.Sleep(400))).
			ToArray();
		Assert.That(NativeProcessRunner.Run("dotnet", "--version").Output, Does.Match(@"\d+\.\d+"));
		Task.WaitAll(busyWorkers);
	}

	[Test]
	public void ToolsShareTheConsoleOfStrict()
	{
		if (!OperatingSystem.IsWindows())
			return;
		var inputEncoding = Console.InputEncoding;
		var outputEncoding = Console.OutputEncoding;
		try
		{
			Console.InputEncoding = Encoding.UTF8;
			Console.OutputEncoding = Encoding.UTF8;
			Assert.That(NativeProcessRunner.Run(Path.Combine(Environment.SystemDirectory, "cmd.exe"),
				"/c chcp").Output, Does.Contain("65001"));
		}
		finally
		{
			Console.InputEncoding = inputEncoding;
			Console.OutputEncoding = outputEncoding;
		}
	}
}
