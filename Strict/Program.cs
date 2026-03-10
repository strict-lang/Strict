using Strict;
using Strict.Bytecode;
using Strict.Expressions;
using Strict.Language;

//ncrunch: no coverage start
if (args.Length == 0)
{
	Console.WriteLine("Usage: Strict <file.strict|file.strictbinary> [diagnostics|decompile]");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.strict diagnostics");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.strictbinary");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.strictbinary decompile");
	return;
}
var filePath = args[0];
if (!File.Exists(filePath))
{
	Console.WriteLine($"Error: File not found: {filePath}");
	Environment.ExitCode = 1;
	return;
}
try
{
	using var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
	if (args.Length > 1 && args[1].Equals("decompile", StringComparison.OrdinalIgnoreCase))
	{
		var outputFolder = Path.GetFileNameWithoutExtension(filePath);
		new BytecodeDecompiler(basePackage).Decompile(filePath, outputFolder);
		Console.WriteLine("Decompilation complete, written all partial .strict files (only what " +
			"was included in bytecode, no tests) to folder: " + outputFolder);
		return;
	}
	var diagnostics = args.Length > 1 &&
		args[1].Equals("diagnostics", StringComparison.OrdinalIgnoreCase);
#if DEBUG
	if (!diagnostics)
		diagnostics = true;
#endif
	new Runner(basePackage, filePath, diagnostics).Run();
}
catch (Exception ex)
{
	Console.WriteLine($"Execution failed: {ex.GetType().Name}: {ex.Message}");
	if (ex.InnerException != null)
		Console.WriteLine($"  Inner: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
	Environment.ExitCode = 1;
}