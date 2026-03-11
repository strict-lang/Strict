namespace Strict.Compiler.X64;

/// <summary>
/// Thrown when a required external tool (NASM, gcc, etc.) cannot be found on PATH.
/// </summary>
public sealed class ToolNotFoundException : Exception
{
	public ToolNotFoundException(string toolName, string downloadUrl)
		: base($"Required tool '{toolName}' was not found on PATH. " +
			$"Download it from: {downloadUrl}") { }
}
