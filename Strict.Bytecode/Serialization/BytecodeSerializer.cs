namespace Strict.Bytecode.Serialization;

/// <summary>
/// Compatibility class providing constants for bytecode file extensions.
/// The actual serialization is done by <see cref="BinaryExecutable"/>.
/// </summary>
public static class BytecodeSerializer
{
	public const string Extension = BinaryExecutable.Extension;
	public const string BytecodeEntryExtension = BinaryType.BytecodeEntryExtension;
	public const byte Version = BinaryType.Version;
}
