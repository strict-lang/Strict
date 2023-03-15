using Strict.Language;

namespace Strict.VirtualMachine;

public sealed class InstanceInvokedMethod : InvokedMethod
{
	public InstanceInvokedMethod(IReadOnlyList<Expression> expressions,
		IReadOnlyDictionary<string, Instance> arguments, Instance instanceCall) : base(expressions,
		arguments) =>
		InstanceCall = instanceCall;

	public Instance InstanceCall { get; }
}