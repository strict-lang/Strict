namespace Strict.Runtime.Statements;

/// <summary>
/// Loads a constant value to one of the <see cref="Register" />s, which could be an actual
/// number, a boolean or a pointer to some memory (usually an offset).
/// </summary>
public sealed class LoadConstantStatement(Register register, Instance constant)
	: InstanceStatement(Instruction.LoadConstantToRegister, constant)
{
	public Register Register { get; } = register;
	public override string ToString() => $"{base.ToString()} {Register}";
}