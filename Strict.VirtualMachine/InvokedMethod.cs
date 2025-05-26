using Strict.Language;

namespace Strict.VirtualMachine;

public class InvokedMethod(IReadOnlyList<Expression> expressions,
	IReadOnlyDictionary<string, Instance> arguments)
{
	public IReadOnlyList<Expression> Expressions { get; } = expressions;
	public IReadOnlyDictionary<string, Instance> Arguments { get; } = arguments;
}