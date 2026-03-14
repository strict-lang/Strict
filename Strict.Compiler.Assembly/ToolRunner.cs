using System.Diagnostics;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Shared helpers for finding external tools on PATH and running processes.
/// Used by both NativeExecutableLinker (NASM+gcc) and LlvmLinker (clang).
/// </summary>
public static class ToolRunner
{
	public static string? FindTool(string name)
	{
		if (!OperatingSystem.IsWindows())
		{ //ncrunch: no coverage start
			try
			{
				var result = RunProcess("which", name);
				if (result.Trim().Length > 0 && File.Exists(result.Trim()))
					return result.Trim();
			}
			catch (InvalidOperationException ex) when (ex.Message.Contains("exit code"))
			{
				// `which` exits non-zero when the tool is not found; fall through to PATH search
			}
		} //ncrunch: no coverage end
		var executableName = OperatingSystem.IsWindows()
			? name + ".exe"
			: name;
		foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(
			Path.PathSeparator))
		{
			var candidate = Path.Combine(dir, executableName);
			if (File.Exists(candidate))
				return candidate; //ncrunch: no coverage
		}
		return null; //ncrunch: no coverage
	}

	//ncrunch: no coverage start
	public static string RunProcess(string executable, string arguments,
		int timeoutMs = DefaultTimeoutMilliseconds)
	{
		using var process = new Process();
		process.StartInfo = new ProcessStartInfo(executable, arguments)
		{
			RedirectStandardOutput = true,
			RedirectStandardError = true,
			UseShellExecute = false,
			CreateNoWindow = true
		};
		process.Start();
		var output = process.StandardOutput.ReadToEnd();
		var error = process.StandardError.ReadToEnd();
		if (!process.WaitForExit(timeoutMs))
		{
			process.Kill();
			throw new InvalidOperationException(
				$"Process '{executable} {arguments}' timed out after {timeoutMs} ms");
		}
		if (process.ExitCode == 0)
			return output;
		var details = string.IsNullOrWhiteSpace(error)
			? output
			: string.IsNullOrWhiteSpace(output)
				? error
				: output + Environment.NewLine + error;
		throw new InvalidOperationException(
			$"Process '{executable} {arguments}' failed with exit code {process.ExitCode}: {details}");
	}

	public static void EnsureOutputFileExists(string outputFilePath, string toolName,
		Platform platform) =>
		_ = ResolveOutputFilePath(outputFilePath, toolName, platform);

	public static string ResolveOutputFilePath(string outputFilePath, string toolName,
		Platform platform)
	{
		if (File.Exists(outputFilePath))
			return outputFilePath;
		if (platform == Platform.Linux && OperatingSystem.IsWindows() &&
			string.Equals(toolName, "gcc", StringComparison.OrdinalIgnoreCase))
		{
			var windowsExecutablePath = outputFilePath + ".exe";
			if (File.Exists(windowsExecutablePath))
				return windowsExecutablePath;
		}
		throw new InvalidOperationException(toolName + " reported success for " + platform +
			" output but did not create file: " + outputFilePath);
	}

	private const int DefaultTimeoutMilliseconds = 30000;
}
