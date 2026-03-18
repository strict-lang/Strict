using Strict.Bytecode;
using Strict.Compiler;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public static class Program
{
	//ncrunch: no coverage start
	public static async Task Main(string[] args)
	{
   args = ResolveImplicitExecutableTarget(args);
		if (args.Length == 0)
			DisplayUsageInformation();
		else
			try
			{
				await ParseArgumentsAndRun(args);
			}
			catch (Exception ex)
			{
				Console.WriteLine($"Execution failed: {ex}");
				Environment.ExitCode = 1;
			}
	}

	private static void DisplayUsageInformation() =>
		Console.WriteLine("""
Usage: Strict <file.strict|.strictbinary> [-options] [args...]

Options (default if nothing specified: build .strictbinary cache and execute in VM)
  -Windows     Compile to a native Windows x64 optimized executable (.exe)
  -Linux       Compile to a native Linux x64 optimized executable
  -MacOS       Compile to a native macOS x64 optimized executable
  -mlir        Force MLIR backend (default, requires mlir-opt + mlir-translate + clang)
               MLIR is the default, best optimized, uses parallel CPU and GPU (Cuda) execution
  -llvm        Force LLVM IR backend (fallback, requires clang: https://releases.llvm.org)
  -nasm        Force NASM backend (fallback, less optimized, requires nasm + gcc/clang)
  -diagnostics Output detailed step-by-step logs and timing for each pipeline stage
               (automatically enabled in Debug builds)
  -decompile   Decompile a .strictbinary into partial .strict source files
               (creates a folder with one .strict per type; no tests, optimized)

Arguments:
  args...      Optional text or numbers passed to called method
               Example to call Run method: Strict Sum.strict 5 10 20 => prints 35
               Example to call any expression, must contain brackets: (1, 2, 3).Length => 3

Examples:
  Strict Examples/SimpleCalculator.strict
  Strict Examples/SimpleCalculator.strict -Windows
  Strict Examples/SimpleCalculator.strict -diagnostics
  Strict Examples/SimpleCalculator.strictbinary
  Strict Examples/SimpleCalculator.strictbinary -decompile
  Strict Examples/Sum.strict 5 10 20
  Strict List.strict (1, 2, 3).Length

Notes:
	Only .strict files contain the full actual code, everything after that is stripped,
	optimized, and just includes what is actually executed (.strictbinary is much smaller).
  Always caches bytecode into a .strictbinary for fast subsequent execution.
  .strictbinary files are reused when they are newer than all of the used source files.
""");

	private static async Task ParseArgumentsAndRun(IReadOnlyList<string> args)
	{
		var filePath = args[0];
		var options =
			new HashSet<string>(args.Skip(1).Where(arg => arg.StartsWith("-", StringComparison.Ordinal)),
				StringComparer.OrdinalIgnoreCase);
		if (options.Contains("-decompile"))
		{
			var outputFolder = Path.GetFileNameWithoutExtension(filePath);
			using var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
			var bytecodeTypes = new BinaryExecutable(filePath, basePackage);
			new Decompiler().Decompile(bytecodeTypes, outputFolder);
			Console.WriteLine("Decompilation complete, written all partial .strict files (only what " +
				"was included in bytecode, no tests) to folder: " + outputFolder);
		}
		else
		{
			var nonFlagArgs = args.Skip(1).Where(arg => !arg.StartsWith("-", StringComparison.Ordinal)).
				ToArray();
			var diagnostics = options.Contains("-diagnostics");
#if DEBUG
			if (!diagnostics)
				diagnostics = true;
#endif
      var expression = nonFlagArgs.Length == 0
				? Method.Run
				: string.Join(" ", nonFlagArgs);
			var runner = new Runner(filePath, null, expression, diagnostics);
			var buildForPlatform = GetPlatformOption(options);
			var backend = options.Contains("-nasm")
				? CompilerBackend.Nasm
				: options.Contains("-llvm")
					? CompilerBackend.Llvm
					: CompilerBackend.MlirDefault;
			if (buildForPlatform.HasValue)
				await runner.Build(buildForPlatform.Value, backend);
			else
				await runner.Run();
		}
	}

	private static string[] ResolveImplicitExecutableTarget(string[] args)
	{
		if (args.Length > 0 && (args[0].EndsWith(Type.Extension, StringComparison.OrdinalIgnoreCase) ||
			args[0].EndsWith(BinaryExecutable.Extension, StringComparison.OrdinalIgnoreCase) ||
			Directory.Exists(args[0])))
			return args;
		var processPath = Environment.ProcessPath;
		if (string.IsNullOrEmpty(processPath))
			return args;
		var implicitBinaryPath = Path.ChangeExtension(processPath, BinaryExecutable.Extension);
		if (File.Exists(implicitBinaryPath))
			return [implicitBinaryPath, .. args];
		var implicitSourcePath = Path.ChangeExtension(processPath, Type.Extension);
		return File.Exists(implicitSourcePath)
			? [implicitSourcePath, .. args]
			: args;
	}

	private static Platform? GetPlatformOption(ICollection<string> options)
	{
		if (options.Contains("-Windows"))
			return Platform.Windows;
		if (options.Contains("-Linux"))
			return Platform.Linux;
		if (options.Contains("-MacOS"))
			return Platform.MacOS;
		return null;
	}
}