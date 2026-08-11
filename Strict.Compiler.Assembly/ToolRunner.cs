namespace Strict.Compiler.Assembly;

/// <summary>
/// Shared helpers for finding external tools on PATH and running processes.
/// Delegates to <see cref="Strict.Expressions.NativeProcessRunner"/> so C# linkers and
/// Strict Process.strict share one implementation.
/// </summary>
public static class ToolRunner
{
	public static string? FindTool(string name) =>
		Strict.Expressions.NativeProcessRunner.FindTool(name);

	public static string RunProcess(string executable, string arguments,
		int timeoutMs = Strict.Expressions.NativeProcessRunner.DefaultTimeoutMilliseconds)
	{
		var result = Strict.Expressions.NativeProcessRunner.Run(executable, arguments, timeoutMs);
		if (result.Succeeded)
			return result.Output;
		var details = string.IsNullOrWhiteSpace(result.Error)
			? result.Output
			: string.IsNullOrWhiteSpace(result.Output)
				? result.Error
				: result.Output + Environment.NewLine + result.Error;
		throw new InvalidOperationException(
			$"Process '{executable} {arguments}' failed with exit code {result.ExitCode}: {details}");
	}

	public static void EnsureOutputFileExists(string outputFilePath, string toolName,
		Platform platform) =>
		_ = ResolveOutputFilePath(outputFilePath, toolName, platform);

	public static string ResolveOutputFilePath(string outputFilePath, string toolName,
		Platform platform)
	{
		if (File.Exists(outputFilePath))
			return outputFilePath;
		if (platform == Platform.Windows ||
			platform == Platform.Linux && OperatingSystem.IsWindows() &&
			string.Equals(toolName, "gcc", StringComparison.OrdinalIgnoreCase))
		{
			var windowsExecutablePath = outputFilePath.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)
				? outputFilePath
				: outputFilePath + ".exe";
			if (File.Exists(windowsExecutablePath))
				return windowsExecutablePath;
		}
		throw new InvalidOperationException(toolName + " reported success for " + platform +
			" output but did not create file: " + outputFilePath);
	}
}
