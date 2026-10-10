using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Strict.Expressions;

/// <summary>
/// Find tools on PATH and run external processes. Shared by the Strict VM (Process.strict)
/// and the C# native linkers so tool invocation is one implementation.
/// </summary>
public static class NativeProcessRunner
{
	public const int DefaultTimeoutMilliseconds = 30000;
	public const string OperatingSystemMethod = "OperatingSystem";
	public static string OperatingSystemName =>
		OperatingSystem.IsWindows()
			? "Windows"
			: OperatingSystem.IsMacOS()
				? "MacOS"
				: "Linux";

	public static string? FindTool(string name)
	{
		if (string.IsNullOrWhiteSpace(name))
			return null;
		if (!OperatingSystem.IsWindows())
			try
			{
				var whichResult = RunCaptured("which", name, 5000);
				if (whichResult.ExitCode == 0)
				{
					var path = whichResult.Output.Trim();
					if (path.Length > 0 && File.Exists(path))
						return path;
				}
			}
			catch
			{
				// fall through to PATH scan
			}
		var executableName = OperatingSystem.IsWindows()
			? name.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)
				? name
				: name + ".exe"
			: name;
		foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator,
			StringSplitOptions.RemoveEmptyEntries))
		{
			var candidate = Path.Combine(dir.Trim('"'), executableName);
			if (File.Exists(candidate))
				return candidate;
		}
		return null;
	}

	public static ProcessRunResult Run(string executable, string arguments,
		int timeoutMs = DefaultTimeoutMilliseconds)
	{
		if (string.IsNullOrWhiteSpace(executable))
			return new ProcessRunResult(127, "", "executable is empty");
		try
		{
			return RunCaptured(ResolveExecutable(executable), arguments, timeoutMs);
		}
		catch (Exception ex)
		{
			return new ProcessRunResult(1, "", ex.Message);
		}
	}

	/// <summary>
	/// Windows searches the application directory before the current one for relative paths.
	/// </summary>
	public static string ResolveExecutable(string executable) =>
		Path.IsPathRooted(executable) || Path.GetFileName(executable) == executable
			? executable
			: Path.GetFullPath(executable);

	private static ProcessRunResult RunCaptured(string executable, string arguments, int timeoutMs)
	{
		using var process = new Process();
		process.StartInfo = new ProcessStartInfo(executable, arguments)
		{
			RedirectStandardOutput = true,
			RedirectStandardError = true,
			UseShellExecute = false,
			CreateNoWindow = OperatingSystem.IsWindows() && GetConsoleCP() == 0
		};
		var output = new StringBuilder();
		var error = new StringBuilder();
		var outputClosed = new TaskCompletionSource();
		var errorClosed = new TaskCompletionSource();
		process.OutputDataReceived += (_, args) => AppendLine(output, outputClosed, args.Data);
		process.ErrorDataReceived += (_, args) => AppendLine(error, errorClosed, args.Data);
		process.Start();
		process.BeginOutputReadLine();
		process.BeginErrorReadLine();
		var exited = process.WaitForExit(timeoutMs);
		if (!exited)
			try
			{
				process.Kill(true);
			}
			catch
			{
				// ignore kill failures
			}
		// ponytail: 2s drain; a grandchild can hold the redirected pipe open forever
		Task.WaitAll([outputClosed.Task, errorClosed.Task], 2000);
		return exited
			? new ProcessRunResult(process.ExitCode, output.ToString(), error.ToString())
			: new ProcessRunResult(124, output.ToString(), "timed out after " + timeoutMs + " ms");
	}

	/// <summary>
	/// Tools share Strict's console, a new hidden console per tool costs ~8 ms on Windows. Only
	/// without any console (0) they need their own, else each one would open a visible window.
	/// </summary>
	[DllImport("kernel32.dll")]
	private static extern uint GetConsoleCP();

	/// <summary>
	/// WaitForExit with a timeout does not wait for the async readers, the null line marks the end.
	/// </summary>
	private static void AppendLine(StringBuilder text, TaskCompletionSource closed, string? line)
	{
		if (line == null)
			closed.TrySetResult();
		else
			text.AppendLine(line);
	}

	public readonly record struct ProcessRunResult(int ExitCode, string Output, string Error)
	{
		public bool Succeeded => ExitCode == 0;
	}
}