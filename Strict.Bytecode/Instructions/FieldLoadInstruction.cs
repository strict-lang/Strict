namespace Strict.Bytecode.Instructions;

/// <summary>
/// Loads a named field value from a ValueTypeInstance stored in <see cref="ObjectRegister"/>
/// into <see cref="RegisterInstruction.Register"/>. Used by the constructor-to-field-mutations
/// optimizer to replace Color.from(obj.Red+x,...) with direct field reads.
/// </summary>
public sealed class FieldLoadInstruction(Register outRegister, Register objectRegister, string fieldName)
  : RegisterInstruction(InstructionType.FieldLoad, outRegister)
{
  public Register ObjectRegister { get; } = objectRegister;
  public string FieldName { get; } = fieldName;
  public override string ToString() => $"{InstructionType} {Register} <- {ObjectRegister}.{FieldName}";
}
