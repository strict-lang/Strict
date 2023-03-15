using Strict.Language;

namespace Strict.VirtualMachine;

public sealed class InstanceInvokedMethod : InvokedMethod
{
	//ncrunch: no coverage start, TODO: missing tests
	public InstanceInvokedMethod(IReadOnlyList<Expression> expressions,
		IReadOnlyDictionary<string, Instance> arguments, Expression instanceCall) : base(expressions,
		arguments) =>
		InstanceCall = instanceCall;

	public Expression InstanceCall { get; }
}