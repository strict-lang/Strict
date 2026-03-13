namespace Strict.Bytecode.Instructions;

/// <summary>
/// Emits a line of text to standard output, optionally appending a value from a register.
/// Replaces the runtime-only Invoke(logger.Log) pattern so the assembly backend can emit printf.
/// </summary>
public sealed class PrintInstruction(string textPrefix, Register? valueRegister = null, bool valueIsText = false)
	: Instruction(InstructionType.Print)
{
	public string TextPrefix { get; } = textPrefix;
	public Register? ValueRegister { get; } = valueRegister;
	public bool ValueIsText { get; } = valueIsText;

	public override string ToString() =>
		ValueRegister.HasValue
			? $"Print \"{
				TextPrefix
			}\" + {
				(ValueIsText
					? "text"
					: "number")
			} {
				ValueRegister.Value
			}"
			: $"Print \"{TextPrefix}\"";
}
