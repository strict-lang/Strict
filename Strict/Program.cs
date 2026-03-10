using Strict;

//ncrunch: no coverage start
if (args.Length == 0)
{
	Console.WriteLine("Usage: Strict <file.strict>");
	Console.WriteLine("Example: Strict Examples/SimpleCalculator.strict");
	return;
}
var strictFilePath = args[0];
if (!File.Exists(strictFilePath))
{
	Console.WriteLine($"Error: File not found: {strictFilePath}");
	Environment.ExitCode = 1;
	return;
}
try
{
	new Runner(strictFilePath,
#if DEBUG
		true
#else
		false
#endif
	).Run();
}
catch (Exception ex)
{
	Console.WriteLine($"Pipeline failed: {ex.GetType().Name}: {ex.Message}");
	if (ex.InnerException != null)
		Console.WriteLine($"  Inner: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
	Environment.ExitCode = 1;
}