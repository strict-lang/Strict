using Strict.Expressions;

namespace Strict.Runtime.Instructions;

public sealed class Invoke(Register register, MethodCall method, Registry persistedRegistry)
	: RegisterInstruction(InstructionType.Invoke, register)
{
	public MethodCall? Method { get; } = method;
	public Registry? PersistedRegistry { get; } = persistedRegistry;
}