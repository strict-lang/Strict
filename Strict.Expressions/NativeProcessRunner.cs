using System.Diagnostics;
using System.Text;

namespace Strict.Expressions;

/// <summary>
/// Find tools on PATH and run external processes. Shared by the Strict VM (Process.strict)
/// and the C# native linkers so tool invocation is one implementation.
/// </summary>
public static class NativeProcessRunner
{
	public const int DefaultTimeoutMilliseconds = 30000;

	public static string? FindTool(string name)
	{
		if (string.IsNullOrWhiteSpace(name))
			return null;
		if (!OperatingSystem.IsWindows())
		{
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
		}
		var executableName = OperatingSystem.IsWindows()
			? name.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)
				? name
				: name + ".exe"
			: name;
		foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(
			Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
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
			return RunCaptured(executable, arguments ?? "", timeoutMs);
		}
		catch (Exception ex)
		{
			return new ProcessRunResult(1, "", ex.Message);
		}
	}

	private static ProcessRunResult RunCaptured(string executable, string arguments, int timeoutMs)
	{
		using var process = new Process();
		process.StartInfo = new ProcessStartInfo(executable, arguments)
		{
			RedirectStandardOutput = true,
			RedirectStandardError = true,
			UseShellExecute = false,
			CreateNoWindow = true
		};
		var output = new StringBuilder();
		var error = new StringBuilder();
		process.OutputDataReceived += (_, args) =>
		{
			if (args.Data != null)
				output.AppendLine(args.Data);
		};
		process.ErrorDataReceived += (_, args) =>
		{
			if (args.Data != null)
				error.AppendLine(args.Data);
		};
		process.Start();
		process.BeginOutputReadLine();
		process.BeginErrorReadLine();
		if (!process.WaitForExit(timeoutMs))
		{
			try
			{
				process.Kill(entireProcessTree: true);
			}
			catch
			{
				// ignore kill failures
			}
			return new ProcessRunResult(124, output.ToString(),
				"timed out after " + timeoutMs + " ms");
		}
		// Ensure async readers finish
		process.WaitForExit();
		return new ProcessRunResult(process.ExitCode, output.ToString(), error.ToString());
	}

	public readonly record struct ProcessRunResult(int ExitCode, string Output, string Error)
	{
		public bool Succeeded => ExitCode == 0;
	}
}
