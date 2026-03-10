using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

public sealed class InstanceInvokedMethod(IReadOnlyList<Expression> expressions,
	IReadOnlyDictionary<string, ValueInstance> arguments,
	ValueInstance instanceCall,
	Type returnType) : InvokedMethod(expressions, arguments, returnType)
{
	public ValueInstance InstanceCall { get; } = instanceCall;
}