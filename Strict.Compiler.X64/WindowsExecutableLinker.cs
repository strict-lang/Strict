using System.Diagnostics;

namespace Strict.Compiler.X64;

/// <summary>
/// Attempts to assemble a NASM .asm file and link it into a Windows .exe using NASM + gcc.
/// Requires: nasm (https://nasm.us) and gcc (MinGW on Windows, system gcc on Linux/macOS).
/// If the required tools are not found, returns null gracefully instead of throwing.
/// </summary>
public sealed class WindowsExecutableLinker
{
	public static bool IsNasmAvailable => FindTool("nasm") != null;
	public static bool IsGccAvailable => FindTool("gcc") != null;

	/// <summary>
	/// Assembles <paramref name="asmPath"/> with NASM then links to a .exe with gcc.
	/// Returns the full path of the .exe on success; null if any tool is missing or fails.
	/// </summary>
	public string? TryCreateExecutable(string asmPath)
	{
		var nasmPath = FindTool("nasm");
		if (nasmPath == null)
			return null;
		var objPath = Path.ChangeExtension(asmPath, ".obj");
		if (!RunProcess(nasmPath, $"-f win64 \"{asmPath}\" -o \"{objPath}\""))
			return null;
		var gccPath = FindTool("gcc");
		if (gccPath == null)
			return null;
		var exePath = Path.ChangeExtension(asmPath, ".exe");
		if (!RunProcess(gccPath, $"\"{objPath}\" -o \"{exePath}\" -lkernel32"))
			return null;
		return File.Exists(exePath) ? exePath : null;
	}

	private static string? FindTool(string name)
	{
		if (!OperatingSystem.IsWindows())
		{
			var result = RunProcessWithOutput("which", name);
			if (result != null && result.Trim().Length > 0 && File.Exists(result.Trim()))
				return result.Trim();
		}
		var windowsName = OperatingSystem.IsWindows() ? name + ".exe" : name;
		foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
		{
			var candidate = Path.Combine(dir, windowsName);
			if (File.Exists(candidate))
				return candidate;
		}
		return null;
	}

	private static bool RunProcess(string executable, string arguments)
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
			return false;
		}
		return process.ExitCode == 0;
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
