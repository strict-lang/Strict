using Strict.Bytecode;
using Strict.Compiler;
using Strict.Expressions;
using Strict.Language;

namespace Strict;

public static class Program
{
	//ncrunch: no coverage start
	public static async Task Main(string[] args)
	{
		if (args.Length == 0)
			DisplayUsageInformation();
		else
			try
			{
				await ParseArgumentsAndRun(args);
			}
			catch (Exception ex)
			{
				Console.WriteLine($"Execution failed: {ex.GetType().Name}: {ex.Message}");
				if (ex.InnerException != null)
					Console.WriteLine($"  Inner: {
						ex.InnerException.GetType().Name
					}: {
						ex.InnerException.Message
					}");
				Environment.ExitCode = 1;
			}
	}

	private static void DisplayUsageInformation()
	{
		Console.WriteLine("Usage: Strict <file.strict|directory|file.strictbinary> [-options] [args...]");
		Console.WriteLine();
		Console.WriteLine("Options:");
		Console.WriteLine("  -diagnostics   Output detailed step-by-step logs and timing for each pipeline stage");
		Console.WriteLine("                 (automatically enabled in Debug builds)");
		Console.WriteLine("  -decompile     Decompile a .strictbinary into partial .strict source files");
		Console.WriteLine("                 (creates a folder with one .strict per type; no tests, optimized)");
		Console.WriteLine("  -Windows       Compile to a native Windows x64 executable (.exe)");
		Console.WriteLine("  -Linux         Compile to a native Linux x64 executable");
		Console.WriteLine("  -MacOS         Compile to a native macOS x64 executable");
		Console.WriteLine("                 Default backend: LLVM (clang) if available, otherwise NASM + gcc/clang");
		Console.WriteLine("  -llvm          Force LLVM IR backend (requires clang: https://releases.llvm.org)");
		Console.WriteLine("  -nasm          Force NASM backend (requires nasm + gcc/clang)");
		Console.WriteLine();
		Console.WriteLine("Arguments:");
		Console.WriteLine("  args...        Optional numbers passed to the program's Run(numbers) method");
		Console.WriteLine("                 Example: Strict Sum.strict 5 10 20  =>  prints 35");
		Console.WriteLine();
		Console.WriteLine("Examples:");
		Console.WriteLine("  Strict Examples/SimpleCalculator.strict");
		Console.WriteLine("  Strict Examples/SimpleCalculator.strict -diagnostics");
		Console.WriteLine("  Strict Examples/SimpleCalculator.strict -Windows");
		Console.WriteLine("  Strict Examples/SimpleCalculator.strictbinary");
		Console.WriteLine("  Strict Examples/SimpleCalculator.strictbinary -decompile");
		Console.WriteLine("  Strict Examples/Sum.strict 5 10 20");
		Console.WriteLine("  Strict Examples/BaseTypesTest");
	}

	private static async Task ParseArgumentsAndRun(string[] args)
	{
		var filePath = args[0];
		var options =
			new HashSet<string>(args.Skip(1).Where(arg => arg.StartsWith("-", StringComparison.Ordinal)),
				StringComparer.OrdinalIgnoreCase);
		using var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		if (options.Contains("-decompile"))
		{
			var outputFolder = Path.GetFileNameWithoutExtension(filePath);
			new BytecodeDecompiler(basePackage).Decompile(filePath, outputFolder);
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
			using var runner = new Runner(basePackage, filePath, diagnostics);
			if (options.Contains("-llvm") && options.Contains("-nasm"))
				throw new InvalidOperationException(
					"Cannot specify both -llvm and -nasm. Choose one backend.");
			if (options.Contains("-llvm"))
				runner.useLlvm = true;
			else if (options.Contains("-nasm"))
				runner.useNasm = true;
			if (nonFlagArgs.Length == 1 && nonFlagArgs[0].Contains('('))
				runner.RunExpression(nonFlagArgs[0]);
			else
				runner.Run(GetPlatformOption(options), nonFlagArgs);
		}
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