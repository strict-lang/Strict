using System.Diagnostics;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles an LLVM IR .ll file directly into a native executable using clang. This replaces the
/// two-step NASM+gcc pipeline with a single clang invocation that includes LLVM's full optimization
/// pipeline (-O2 by default), producing smaller and faster executables.
/// Requires clang on PATH (https://releases.llvm.org or via platform package manager).
/// </summary>
public sealed class LlvmLinker
{
	/// <summary>
	/// Compiles <paramref name="llvmIrPath"/> (.ll file) directly to a native executable.
	/// Clang runs LLVM optimization passes and handles platform-specific linking automatically.
	/// </summary>
	public string CreateExecutable(string llvmIrPath, Platform platform)
	{
		var clangPath = FindTool("clang") ??
			throw new ToolNotFoundException("clang", DownloadUrlFor(platform));
		var exeExtension = platform == Platform.Windows
			? ".exe"
			: "";
		var exeFilePath = Path.ChangeExtension(llvmIrPath, null) + exeExtension;
		var arguments = BuildClangArgs(llvmIrPath, exeFilePath, platform);
		RunProcess(clangPath, arguments);
		EnsureOutputFileExists(exeFilePath, "clang", platform);
		return exeFilePath;
	}

	public static bool IsClangAvailable => FindTool("clang") != null;

	private static string BuildClangArgs(string inputPath, string outputPath, Platform platform) =>
		platform switch
		{
			Platform.Windows =>
				$"\"{inputPath}\" -o \"{outputPath}\" -O2 -Wno-override-module",
			Platform.Linux =>
				$"\"{inputPath}\" -o \"{outputPath}\" -O2 -Wno-override-module",
			Platform.MacOS =>
				$"\"{inputPath}\" -o \"{outputPath}\" -O2 -Wno-override-module",
			_ => throw new NotSupportedException("Unsupported platform: " + platform)
		};

	//ncrunch: no coverage start
	private static string DownloadUrlFor(Platform platform) =>
		platform switch
		{
			Platform.Windows => "https://releases.llvm.org",
			Platform.Linux => "https://apt.llvm.org",
			Platform.MacOS => "https://developer.apple.com/xcode",
			_ => "https://releases.llvm.org"
		}; //ncrunch: no coverage end

	private static string? FindTool(string name)
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
	private static string RunProcess(string executable, string arguments)
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
		if (!process.WaitForExit(TimeoutMilliseconds))
		{
			process.Kill();
			throw new InvalidOperationException(
				$"Process '{executable} {arguments}' timed out after {TimeoutMilliseconds} ms");
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

	private static void EnsureOutputFileExists(string outputFilePath, string toolName,
		Platform platform)
	{
		if (!File.Exists(outputFilePath))
			throw new InvalidOperationException(toolName + " reported success for " + platform +
				" output but did not create file: " + outputFilePath);
	}

	private const int TimeoutMilliseconds = 30000;
}
