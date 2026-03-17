namespace Strict.Compiler;

public abstract class Linker
{
	public abstract Task<string> CreateExecutable(string asmFilePath, Platform platform,
		bool hasPrintCalls = false);
}