using System.Diagnostics;

namespace Strict.Compiler.X64;

/// <summary>
/// Assembles a NASM .asm file and links it into a native executable for the requested platform.
/// Requires NASM (https://nasm.us) and gcc/clang on PATH.
/// Throws <see cref="ToolNotFoundException"/> with download instructions when tools are missing.
/// </summary>
public sealed class NativeExecutableLinker
{
	public static bool IsNasmAvailable => FindTool("nasm") != null;
	public static bool IsGccAvailable => FindTool("gcc") != null;

	/// <summary>
	/// Assembles <paramref name="asmPath"/> with NASM and links it into an executable.
	/// Throws <see cref="ToolNotFoundException"/> if NASM or the C compiler is not on PATH.
	/// Throws <see cref="NotSupportedException"/> for platforms whose code generation is not yet
	/// implemented (e.g., LinuxArm requires a separate AArch64 code-generator).
	/// </summary>
	public string CreateExecutable(string asmPath, Platform platform)
	{
		if (platform == Platform.LinuxArm)
			throw new NotSupportedException(
				"AArch64 code generation is not yet implemented. " +
				"A dedicated ARM code-generator is required for LinuxArm.");
		var nasmPath = FindTool("nasm") ??
			throw new ToolNotFoundException("nasm", "https://nasm.us");
		var objPath = Path.ChangeExtension(asmPath, ".obj");
		var nasmFormat = NasmFormatFor(platform);
		RunProcess(nasmPath, $"-f {nasmFormat} \"{asmPath}\" -o \"{objPath}\"");
		var linker = platform == Platform.MacOS ? "clang" : "gcc";
		var linkerPath = FindTool(linker) ??
			throw new ToolNotFoundException(linker, LinkerDownloadUrlFor(platform));
		var exeExtension = platform == Platform.Windows ? ".exe" : "";
		var exePath = Path.ChangeExtension(asmPath, null) + exeExtension;
		var linkerArgs = BuildLinkerArgs(objPath, exePath, platform);
		RunProcess(linkerPath, linkerArgs);
		return exePath;
	}

	private static string NasmFormatFor(Platform platform) =>
		platform switch
		{
			Platform.Windows => "win64",
			Platform.Linux => "elf64",
			Platform.MacOS => "macho64",
			_ => throw new NotSupportedException("Unsupported platform: " + platform)
		};

	private static string BuildLinkerArgs(string objPath, string exePath, Platform platform) =>
		platform switch
		{
			Platform.Windows => $"\"{objPath}\" -o \"{exePath}\" -lkernel32",
			Platform.Linux => $"\"{objPath}\" -o \"{exePath}\"",
			Platform.MacOS => $"\"{objPath}\" -o \"{exePath}\"",
			_ => throw new NotSupportedException("Unsupported platform: " + platform)
		};

	private static string LinkerDownloadUrlFor(Platform platform) =>
		platform switch
		{
			Platform.Windows => "https://www.mingw-w64.org",
			Platform.Linux => "https://gcc.gnu.org",
			Platform.MacOS => "https://developer.apple.com/xcode",
			_ => "https://gcc.gnu.org"
		};

	private static string? FindTool(string name)
	{
		if (!OperatingSystem.IsWindows())
		{
			var result = RunProcessWithOutput("which", name);
			if (result != null && result.Trim().Length > 0 && File.Exists(result.Trim()))
				return result.Trim();
		}
		var executableName = OperatingSystem.IsWindows() ? name + ".exe" : name;
		foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
		{
			var candidate = Path.Combine(dir, executableName);
			if (File.Exists(candidate))
				return candidate;
		}
		return null;
	}

	private static void RunProcess(string executable, string arguments)
	{
		using var process = new Process
		{
			StartInfo = new ProcessStartInfo(executable, arguments)
			{
				RedirectStandardOutput = true,
				RedirectStandardError = true,
				UseShellExecute = false,
				CreateNoWindow = true
			}
		};
		process.Start();
		if (!process.WaitForExit(TimeoutMilliseconds))
		{
			process.Kill();
			throw new InvalidOperationException($"Process '{executable}' timed out after {TimeoutMilliseconds} ms");
		}
		if (process.ExitCode != 0)
		{
			var error = process.StandardError.ReadToEnd();
			throw new InvalidOperationException(
				$"Process '{executable}' failed with exit code {process.ExitCode}: {error}");
		}
	}

	private static string? RunProcessWithOutput(string executable, string arguments)
	{
		try
		{
			using var process = new Process
			{
				StartInfo = new ProcessStartInfo(executable, arguments)
				{
					RedirectStandardOutput = true,
					RedirectStandardError = true,
					UseShellExecute = false,
					CreateNoWindow = true
				}
			};
			process.Start();
			var output = process.StandardOutput.ReadToEnd();
			if (!process.WaitForExit(TimeoutMilliseconds))
			{
				process.Kill();
				return null;
			}
			return process.ExitCode == 0 ? output : null;
		}
		catch (Exception)
		{
			return null;
		}
	}

	private const int TimeoutMilliseconds = 10000;
}
