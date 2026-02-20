using Strict.Expressions;

namespace Strict.Runtime.Statements;

public sealed class Invoke : RegisterStatement
{
	public Invoke(MethodCall method, Register register, Registry persistedRegistry)
		: base(Instruction.Invoke, register)
	{
		Method = method;
		PersistedRegistry = persistedRegistry;
	}

	// Used for test comparisons only (ToString gives same result as the full constructor)
	public Invoke(string _, Register register) : base(Instruction.Invoke, register) { }

	public MethodCall? Method { get; }
	public Registry? PersistedRegistry { get; }
}