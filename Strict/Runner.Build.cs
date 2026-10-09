// When things are in flux, force generating a new .strictbinary every time by disabling the cache
//#define DISABLE_BINARY_CACHE
using System.Globalization;
using Strict.Bytecode;
using Strict.Bytecode.Serialization;
using Strict.Compiler;
using Strict.Compiler.Assembly;
using Strict.Expressions;
using Strict.Language;
using Strict.Optimizers;
using Strict.TestRunner;
using Strict.Validators;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class Runner
{
	/// <summary>
	/// Generates a platform-specific executable from the compiled instructions. Uses MLIR when
	/// -mlir is specified, LLVM IR when -llvm is specified, otherwise NASM + gcc/clang pipeline.
	/// Throws <see cref="ToolNotFoundException"/> if required tools are missing.
	/// </summary>
	public async Task Build(Platform platform, CompilerBackend backend = CompilerBackend.MlirDefault)
	{
		if (IsExpressionInvocation)
			throw new CannotBuildExecutableWithCustomExpression();
		var binary = await GetBinary();
		if (binary.GetRunMethods().Any(method => method.parameters.Count > 0))
		{
			var launcherPath = CreateManagedLauncher(platform);
			PrintLauncherSummary(platform, launcherPath);
			return;
		}
		InstructionsCompiler compiler = backend switch
		{
			CompilerBackend.Llvm => new InstructionsToLlvmIr(),
			CompilerBackend.Nasm => new InstructionsToAssembly(),
			_ => new InstructionsToMlir()
		};
		Linker linker = backend switch
		{
			CompilerBackend.Llvm => new LlvmLinker(),
			CompilerBackend.Nasm => new NativeExecutableLinker(),
			_ => new MlirLinker()
		};
		var irFilePath = Path.ChangeExtension(strictFilePath, compiler.Extension);
		await File.WriteAllTextAsync(irFilePath, await compiler.Compile(binary, platform));
		var exeFilePath = await linker.CreateExecutable(irFilePath, platform, binary.UsesConsolePrint);
		PrintCompilationSummary(backend, platform, exeFilePath);
	}

	private string CreateManagedLauncher(Platform platform)
	{
		if ((platform == Platform.Windows && !OperatingSystem.IsWindows()) ||
			(platform == Platform.Linux && !OperatingSystem.IsLinux()) ||
			(platform == Platform.MacOS && !OperatingSystem.IsMacOS()))
			throw new BuildRequiresTargetPlatform(platform);
		var runtimeDirectory = Path.GetDirectoryName(typeof(Program).Assembly.Location) ??
			throw new DirectoryNotFoundException("Strict runtime output directory not found.");
		var outputDirectory = Path.GetDirectoryName(Path.GetFullPath(strictFilePath)) ??
			throw new DirectoryNotFoundException("Output directory not found.");
		var runtimeExecutableName = OperatingSystem.IsWindows()
			? "Strict.exe"
			: "Strict";
		var outputExecutablePath = Path.Combine(outputDirectory, OperatingSystem.IsWindows()
			? Path.GetFileNameWithoutExtension(strictFilePath) + ".exe"
			: Path.GetFileNameWithoutExtension(strictFilePath));
		File.Copy(Path.Combine(runtimeDirectory, runtimeExecutableName), outputExecutablePath, true);
		foreach (var filePath in Directory.GetFiles(runtimeDirectory, "*.dll"))
			File.Copy(filePath, Path.Combine(outputDirectory, Path.GetFileName(filePath)), true);
		foreach (var filePath in Directory.GetFiles(runtimeDirectory, "*.json"))
			File.Copy(filePath, Path.Combine(outputDirectory, Path.GetFileName(filePath)), true);
		if (!OperatingSystem.IsWindows())
			File.SetUnixFileMode(outputExecutablePath,
				UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
				UnixFileMode.GroupRead | UnixFileMode.GroupExecute | UnixFileMode.OtherRead |
				UnixFileMode.OtherExecute);
		return outputExecutablePath;
	}

	private static void PrintLauncherSummary(Platform platform, string exeFilePath) =>
		Console.WriteLine("Created " + platform + " executable launcher of " +
			new FileInfo(exeFilePath).Length + " bytes to: " + exeFilePath);

	public sealed class BuildRequiresTargetPlatform(Platform platform)
		: Exception("Runtime launcher builds for " + platform + " require building on that platform.");
}
