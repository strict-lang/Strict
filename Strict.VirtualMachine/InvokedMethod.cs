using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

public class InvokedMethod(IReadOnlyList<Expression> expressions,
	IReadOnlyDictionary<string, Instance> arguments, Type returnType)
{
	public IReadOnlyList<Expression> Expressions { get; } = expressions;
	public IReadOnlyDictionary<string, Instance> Arguments { get; } = arguments;
	public Type ReturnType { get; } = returnType;
}