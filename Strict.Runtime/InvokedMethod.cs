using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

public class InvokedMethod(IReadOnlyList<Expression> expressions,
	IReadOnlyDictionary<string, ValueInstance> arguments, Type returnType)
{
	public IReadOnlyList<Expression> Expressions { get; } = expressions;
	public IReadOnlyDictionary<string, ValueInstance> Arguments { get; } = arguments;
	public Type ReturnType { get; } = returnType;
}