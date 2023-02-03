using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class InvokeStatement : RegisterStatement
{
	public InvokeStatement(MethodCall methodCall, Register register) : base(register,
		Instruction.Invoke)
	{
		MethodCall = methodCall;
		MethodCallText = methodCall.ToString();
	}

	public InvokeStatement(string methodCall, Register register) : base(register,
		Instruction.Invoke) =>
		MethodCallText = methodCall;

	public string MethodCallText { get; } // For tests ONLY
	public MethodCall? MethodCall { get; }
}