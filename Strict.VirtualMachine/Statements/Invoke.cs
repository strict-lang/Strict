using Strict.Expressions;

namespace Strict.Runtime.Statements;

public sealed class Invoke(Register register, MethodCall method, Registry persistedRegistry)
	: RegisterStatement(Instruction.Invoke, register)
{
	public MethodCall? Method { get; } = method;
	public Registry? PersistedRegistry { get; } = persistedRegistry;
}