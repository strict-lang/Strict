using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

public sealed class InstanceInvokedMethod(IReadOnlyList<Expression> expressions,
	IReadOnlyDictionary<string, Instance> arguments, Instance instanceCall, Type returnType) :
	InvokedMethod(expressions, arguments, returnType)
{
	public Instance InstanceCall { get; } = instanceCall;
}