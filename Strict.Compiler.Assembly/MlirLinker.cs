namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles an MLIR .mlir file into a native executable via a three-step pipeline:
///   1. mlir-opt: Lower arith+func dialects to LLVM dialect
///   2. mlir-translate: Convert LLVM dialect to LLVM IR (.ll)
///   3. clang: Compile LLVM IR to native executable with -O2
/// Requires mlir-opt, mlir-translate, and clang on PATH.
/// Falls back gracefully when MLIR tools are unavailable — use IsAvailable to check.
/// </summary>
public sealed class MlirLinker
{
	public string CreateExecutable(string mlirPath, Platform platform)
	{
		var mlirOptPath = ToolRunner.FindTool("mlir-opt") ??
			throw new ToolNotFoundException("mlir-opt",
				"https://github.com/llvm/llvm-project/releases (install MLIR tools or " +
				"on Windows use Msys2 and 'pacman -S mingw-w64-x86_64-mlir')");
		var mlirTranslatePath = ToolRunner.FindTool("mlir-translate") ??
			throw new ToolNotFoundException("mlir-translate",
				"https://github.com/llvm/llvm-project/releases (install MLIR tools or " +
				"on Windows use Msys2 and 'pacman -S mingw-w64-x86_64-mlir')");
		var clangPath = ToolRunner.FindTool("clang") ??
			throw new ToolNotFoundException("clang", "https://releases.llvm.org");
		var llvmDialectPath = Path.ChangeExtension(mlirPath, ".llvm.mlir");
		ToolRunner.RunProcess(mlirOptPath,
			$"\"{mlirPath}\" --convert-arith-to-llvm --convert-func-to-llvm " +
			$"--convert-cf-to-llvm --reconcile-unrealized-casts -o \"{llvmDialectPath}\"");
		ToolRunner.EnsureOutputFileExists(llvmDialectPath, "mlir-opt", platform);
		var llvmIrPath = Path.ChangeExtension(mlirPath, ".ll");
		ToolRunner.RunProcess(mlirTranslatePath,
			$"--mlir-to-llvmir \"{llvmDialectPath}\" -o \"{llvmIrPath}\"");
		ToolRunner.EnsureOutputFileExists(llvmIrPath, "mlir-translate", platform);
		var exeExtension = platform == Platform.Windows
			? ".exe"
			: "";
		var exeFilePath = Path.ChangeExtension(mlirPath, null) + exeExtension;
		ToolRunner.RunProcess(clangPath,
			$"\"{llvmIrPath}\" -o \"{exeFilePath}\" -O2 -Wno-override-module");
		ToolRunner.EnsureOutputFileExists(exeFilePath, "clang", platform);
		return exeFilePath;
	}

	public static bool IsAvailable =>
		ToolRunner.FindTool("mlir-opt") != null &&
		ToolRunner.FindTool("mlir-translate") != null &&
		ToolRunner.FindTool("clang") != null;
}
