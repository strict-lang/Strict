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
		var clangPath = ToolRunner.FindTool("clang") ??
			throw new ToolNotFoundException("clang", DownloadUrlFor(platform));
		var exeExtension = platform == Platform.Windows
			? ".exe"
			: "";
		var exeFilePath = Path.ChangeExtension(llvmIrPath, null) + exeExtension;
		var arguments = BuildClangArgs(llvmIrPath, exeFilePath, platform);
		ToolRunner.RunProcess(clangPath, arguments);
		ToolRunner.EnsureOutputFileExists(exeFilePath, "clang", platform);
		return exeFilePath;
	}

	public static bool IsClangAvailable => ToolRunner.FindTool("clang") != null;

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
}
