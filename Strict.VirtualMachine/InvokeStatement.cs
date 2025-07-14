using Strict.Expressions;

namespace Strict.Runtime;

public sealed class InvokeStatement : RegisterStatement
{
	public InvokeStatement(MethodCall methodCall, Register register, Registry persistedRegistry) :
		base(register, Instruction.Invoke)
	{
		MethodCall = methodCall;
		MethodCallText = methodCall.ToString();
		PersistedRegistry = persistedRegistry;
	}

	//ncrunch: no coverage start, For tests only
	public InvokeStatement(string methodCall, Register register) :
		base(register, Instruction.Invoke) =>
		MethodCallText = methodCall;

	public string MethodCallText { get; } // For tests ONLY
	//ncrunch: no coverage end
	public MethodCall? MethodCall { get; }
	public Registry? PersistedRegistry { get; }
}