namespace Strict.Compiler;

/// <summary>
/// Thrown when a required external tool (NASM, gcc, etc.) cannot be found on PATH.
/// </summary>
public sealed class ToolNotFoundException(string toolName, string downloadUrl) : Exception(
	$"Required tool '{toolName}' was not found on PATH. " + $"Download it from: {downloadUrl}");