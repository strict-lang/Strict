using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class Memory
{
	public Dictionary<Register, Instance> Registers { get; init; } = new();
	public Dictionary<string, Instance> Variables = new();

	public void AddToCollectionVariable(string key, Expression element)
	{
		Variables.TryGetValue(key, out var collection);
		if (collection?.Value is List<Expression> listExpression)
			listExpression.Add(element);
	}
}