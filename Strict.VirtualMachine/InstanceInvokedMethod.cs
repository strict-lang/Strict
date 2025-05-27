using Strict.Language;

namespace Strict.VirtualMachine;

public sealed class InstanceInvokedMethod(IReadOnlyList<Expression> expressions,
	IReadOnlyDictionary<string, Instance> arguments, Instance instanceCall) :
	InvokedMethod(expressions, arguments)
{
	public Instance InstanceCall { get; } = instanceCall;
}