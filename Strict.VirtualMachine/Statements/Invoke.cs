using Strict.Expressions;

namespace Strict.Runtime.Statements;

public sealed class Invoke(MethodCall method, Register register, Registry persistedRegistry)
	: RegisterStatement(Instruction.Invoke, register)
{
	// Used for test comparisons only (ToString gives same result as the full constructor)
	public Invoke(string _, Register register) : this(null!, register, null!) { }

	public MethodCall? Method { get; } = method;
	public Registry? PersistedRegistry { get; } = persistedRegistry;
}