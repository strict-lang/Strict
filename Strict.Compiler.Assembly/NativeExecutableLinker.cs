using System.Diagnostics;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Assembles a NASM .asm file and links it into a native executable for the requested platform.
/// Requires NASM (https://nasm.us) and gcc/clang on PATH.
/// Throws <see cref="ToolNotFoundException"/> with download instructions when tools are missing.
/// </summary>
public sealed class NativeExecutableLinker
{
	/// <summary>
	/// Assembles <paramref name="asmPath"/> with NASM and links it into an executable.
	/// Throws <see cref="ToolNotFoundException"/> if NASM or the C compiler is not on PATH.
	/// Throws <see cref="NotSupportedException"/> for platforms whose code generation is not yet
	/// implemented (e.g., LinuxArm requires a separate AArch64 code-generator).
	/// </summary>
	public string CreateExecutable(string asmPath, Platform platform, bool hasPrintCalls = false)
	{
		var nasmPath = FindTool("nasm") ??
			throw new ToolNotFoundException("nasm", "https://nasm.us");
		var objPath = Path.ChangeExtension(asmPath, ".obj");
		var nasmFormat = NasmFormatFor(platform);
		RunProcess(nasmPath, $"-f {nasmFormat} \"{asmPath}\" -o \"{objPath}\"");
		EnsureOutputFileExists(objPath, "nasm", platform);
		var linker = platform == Platform.MacOS
			? "clang"
			: "gcc";
		var linkerPath = FindTool(linker) ??
			throw new ToolNotFoundException(linker, LinkerDownloadUrlFor(platform));
		var exeExtension = platform == Platform.Windows
			? ".exe"
			: "";
		var exeFilePath = Path.ChangeExtension(asmPath, null) + exeExtension;
		var linkerArgs = BuildLinkerArgs(objPath, exeFilePath, platform, hasPrintCalls);
		RunProcess(linkerPath, linkerArgs);
		EnsureOutputFileExists(exeFilePath, linker, platform);
		return exeFilePath;
	} //ncrunch: no coverage end

	public static bool IsNasmAvailable => FindTool("nasm") != null;
	public static bool IsGccAvailable => FindTool("gcc") != null;

	//ncrunch: no coverage start
	private static string NasmFormatFor(Platform platform) =>
		platform switch
		{
			Platform.Windows => "win64",
			Platform.Linux => "elf64",
			Platform.MacOS => "macho64",
			_ => throw new NotSupportedException("Unsupported platform: " + platform)
		}; //ncrunch: no coverage end

	private static string BuildLinkerArgs(string objPath, string exePath, Platform platform, bool hasPrintCalls = false)
	{
		const string SizeFlags = "-s -Wl,--gc-sections -Wl,--strip-all";
		return platform switch
		{
			Platform.Windows => $"\"{objPath}\" -o \"{exePath}\" {SizeFlags} -nostdlib -Wl,-e,main -lkernel32",
			Platform.Linux => hasPrintCalls //ncrunch: no coverage
				? $"\"{objPath}\" -o \"{exePath}\" {SizeFlags}"
				: $"\"{objPath}\" -o \"{exePath}\" {SizeFlags} -nostdlib -Wl,-e,_start",
			Platform.MacOS => $"\"{objPath}\" -o \"{exePath}\" {SizeFlags}", //ncrunch: no coverage
			_ => throw new NotSupportedException("Unsupported platform: " + platform) //ncrunch: no coverage
		};
	}

	//ncrunch: no coverage start
	private static string LinkerDownloadUrlFor(Platform platform) =>
		platform switch
		{
			Platform.Windows => "https://www.mingw-w64.org",
			Platform.Linux => "https://gcc.gnu.org",
			Platform.MacOS => "https://developer.apple.com/xcode",
			_ => "https://gcc.gnu.org"
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
			throw new InvalidOperationException($"Process '{executable} {arguments}' timed out after {TimeoutMilliseconds} ms");
		}
		if (process.ExitCode == 0)
			return output;
		var details = string.IsNullOrWhiteSpace(error)
			? output
			: string.IsNullOrWhiteSpace(output)
				? error
				: output + Environment.NewLine + error;
		throw new InvalidOperationException($"Process '{executable} {arguments}' failed with exit code {process.ExitCode}: {details}");
	}

	private static void EnsureOutputFileExists(string outputFilePath, string toolName, Platform platform)
	{
		if (!File.Exists(outputFilePath))
			throw new InvalidOperationException(toolName + " reported success for " + platform +
				" output but did not create file: " + outputFilePath);
	}

	private const int TimeoutMilliseconds = 10000;
}