using Strict.Language;

namespace Strict.VirtualMachine;

public sealed class InvokedMethod
{
	public InvokedMethod(IReadOnlyList<Expression> expressions, IReadOnlyDictionary<string, Instance> arguments)
	{
		Expressions = expressions;
		Arguments = arguments;
	}

	public IReadOnlyList<Expression> Expressions { get; }
	public IReadOnlyDictionary<string, Instance> Arguments { get; }
}