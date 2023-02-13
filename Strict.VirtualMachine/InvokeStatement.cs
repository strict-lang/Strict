using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class InvokeStatement : RegisterStatement
{
	public InvokeStatement(MethodCall methodCall, Register register, Registry persistedRegistry) : base(register,
		Instruction.Invoke)
	{
		MethodCall = methodCall;
		MethodCallText = methodCall.ToString();
		PersistedRegistry = persistedRegistry;
	}

	public InvokeStatement(string methodCall, Register register) : base(register,
		Instruction.Invoke) =>
		MethodCallText = methodCall;

	public string MethodCallText { get; } // For tests ONLY
	public MethodCall? MethodCall { get; }
	public Registry? PersistedRegistry { get; }
}