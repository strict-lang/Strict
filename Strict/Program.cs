using Strict;
using Strict.Bytecode;

//ncrunch: no coverage start
if (args.Length == 0)
{
	Console.WriteLine("Usage: Strict <file.strict|file.sbc>");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.strict");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.sbc");
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
	var isBytecodeFile = filePath.EndsWith(BytecodeSerializer.Extension,
		StringComparison.OrdinalIgnoreCase);
	if (isBytecodeFile)
		Runner.LoadBytecodeFile(filePath
#if DEBUG
			, true
#endif
		).Run();
	else
		new Runner(filePath
#if DEBUG
			, true
#endif
		).Run();
}
catch (Exception ex)
{
	Console.WriteLine($"Execution failed: {ex.GetType().Name}: {ex.Message}");
	if (ex.InnerException != null)
		Console.WriteLine($"  Inner: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
	Environment.ExitCode = 1;
}