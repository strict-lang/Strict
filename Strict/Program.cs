using Strict;
using Strict.Bytecode;

//ncrunch: no coverage start
if (args.Length == 0)
{
	Console.WriteLine("Usage: Strict <file.strict|file.strict_binary> [diagnostics]");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.strict diagnostics");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.strict_binary");
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
	var diagnostics = args.Length > 1 &&
		args[1].Equals("diagnostics", StringComparison.OrdinalIgnoreCase);
#if DEBUG
	diagnostics = true;
#endif
	var isBytecodeFile = filePath.EndsWith(BytecodeSerializer.Extension,
		StringComparison.OrdinalIgnoreCase);
	if (isBytecodeFile)
		Runner.LoadBytecodeFile(filePath, diagnostics).Run();
	else
		new Runner(filePath, diagnostics).Run();
}
catch (Exception ex)
{
	Console.WriteLine($"Execution failed: {ex.GetType().Name}: {ex.Message}");
	if (ex.InnerException != null)
		Console.WriteLine($"  Inner: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
	Environment.ExitCode = 1;
}