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
	public string CreateExecutable(string llvmIrPath, Platform platform, bool hasPrintCalls = false)
	{ //ncrunch: no coverage start
		var clangPath = ToolRunner.FindTool("clang") ??
			throw new ToolNotFoundException("clang", DownloadUrlFor(platform));
		var exeExtension = platform == Platform.Windows
			? ".exe"
			: "";
		var exeFilePath = Path.ChangeExtension(llvmIrPath, null) + exeExtension;
		var arguments = BuildClangArgs(llvmIrPath, exeFilePath, platform, hasPrintCalls);
		ToolRunner.RunProcess(clangPath, arguments);
		ToolRunner.EnsureOutputFileExists(exeFilePath, "clang", platform);
		return exeFilePath;
	} //ncrunch: no coverage end

	public static bool IsClangAvailable => ToolRunner.FindTool("clang") != null;

	private static string BuildClangArgs(string inputPath, string outputPath, Platform platform,
		bool hasPrintCalls = false)
	{
		var quotedInputPath = $"\"{inputPath}\"";
		var quotedOutputPath = $"\"{outputPath}\"";
		return platform switch
		{
			Platform.Windows =>
				$"{quotedInputPath} -o {quotedOutputPath} -Oz -nostdlib -lkernel32 -Wl,/ENTRY:main " +
				$"-Wl,/SUBSYSTEM:CONSOLE -Wl,/OPT:REF -Wl,/OPT:ICF -Wl,/INCREMENTAL:NO -Wl,/DEBUG:NONE " +
				$"-Wno-override-module",
			Platform.Linux when OperatingSystem.IsWindows() =>
				$"{quotedInputPath} -o {quotedOutputPath} -Oz -Wno-override-module",
			//ncrunch: no coverage start
			Platform.Linux => hasPrintCalls
				? $"{quotedInputPath} -o {quotedOutputPath} -Oz -s -Wl,--gc-sections -Wl,--strip-all " +
				$"-Wno-override-module"
				: $"{quotedInputPath} -o {quotedOutputPath} -Oz -s -nostdlib -Wl,-e,main " +
				$"-Wl,--gc-sections -Wl,--strip-all -Wno-override-module",
			Platform.MacOS => $"{quotedInputPath} -o {quotedOutputPath} -Oz -Wl,-dead_strip " +
				$"-Wno-override-module",
			_ => throw new NotSupportedException("Unsupported platform: " + platform)
		}; //ncrunch: no coverage end
	}

	//ncrunch: no coverage start
	private static string DownloadUrlFor(Platform platform) =>
		platform switch
		{
			Platform.Windows => "https://releases.llvm.org",
			Platform.Linux => "https://apt.llvm.org",
			Platform.MacOS => "https://developer.apple.com/xcode",
			_ => "https://releases.llvm.org"
		}; //ncrunch: no coverage end
}
