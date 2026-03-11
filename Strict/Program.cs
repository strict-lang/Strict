using Strict;
using Strict.Bytecode;
using Strict.Compiler.X64;
using Strict.Expressions;
using Strict.Language;

//ncrunch: no coverage start
if (args.Length == 0)
{
	Console.WriteLine("Usage: Strict <file.strict|file.strictbinary> [-options] [args...]");
	Console.WriteLine();
	Console.WriteLine("Options:");
	Console.WriteLine("  -diagnostics   Output detailed step-by-step logs and timing for each pipeline stage");
	Console.WriteLine("                 (automatically enabled in Debug builds)");
	Console.WriteLine("  -decompile     Decompile a .strictbinary into partial .strict source files");
	Console.WriteLine("                 (creates a folder with one .strict per type; no tests, optimized)");
	Console.WriteLine("  -Windows       Compile to a native Windows x64 executable (.exe)");
	Console.WriteLine("                 Requires: nasm (https://nasm.us) and gcc (https://www.mingw-w64.org)");
	Console.WriteLine("  -Linux         Compile to a native Linux x64 executable");
	Console.WriteLine("                 Requires: nasm (https://nasm.us) and gcc");
	Console.WriteLine("  -LinuxArm      Compile to a native Linux AArch64 executable (Raspberry Pi, Jetson, …)");
	Console.WriteLine("                 Note: AArch64 code generation is not yet implemented");
	Console.WriteLine("  -MacOS         Compile to a native macOS x64 executable");
	Console.WriteLine("                 Requires: nasm (https://nasm.us) and clang");
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
	return;
}
var filePath = args[0];
var options = new HashSet<string>(args.Skip(1).Where(arg => arg.StartsWith("-", StringComparison.Ordinal)), StringComparer.OrdinalIgnoreCase);
var programArgs = args.Skip(1).Where(arg => !arg.StartsWith("-", StringComparison.Ordinal)).ToArray();
try
{
	using var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
	if (options.Contains("-decompile"))
	{
		var outputFolder = Path.GetFileNameWithoutExtension(filePath);
		new BytecodeDecompiler(basePackage).Decompile(filePath, outputFolder);
		Console.WriteLine("Decompilation complete, written all partial .strict files (only what " +
			"was included in bytecode, no tests) to folder: " + outputFolder);
		return;
	}
	var diagnostics = options.Contains("-diagnostics");
#if DEBUG
	diagnostics = true;
#endif
	var targetPlatform = ParsePlatformOption(options);
	var runArgs = programArgs.Length > 0 ? programArgs : null;
	new Runner(basePackage, filePath, diagnostics).Run(targetPlatform, runArgs);
}
catch (Exception ex)
{
	Console.WriteLine($"Execution failed: {ex.GetType().Name}: {ex.Message}");
	if (ex.InnerException != null)
		Console.WriteLine($"  Inner: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
	Environment.ExitCode = 1;
}

static Platform? ParsePlatformOption(ICollection<string> options)
{
	if (options.Contains("-Windows"))
		return Platform.Windows;
	if (options.Contains("-Linux"))
		return Platform.Linux;
	if (options.Contains("-LinuxArm"))
		return Platform.LinuxArm;
	if (options.Contains("-MacOS"))
		return Platform.MacOS;
	return null;
}
