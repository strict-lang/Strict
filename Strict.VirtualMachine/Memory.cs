using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class Memory
{
	public Dictionary<Register, Instance> Registers { get; init; } = new();
	public Dictionary<string, Instance> Variables = new();

	public void AddToCollectionVariable(string key, object element)
	{
		Variables.TryGetValue(key, out var collection);
		if (collection?.Value is List<Expression> listExpression)
			listExpression.Add(ConvertObjectToValueForm(element, listExpression[0]));
	}

	private static Value ConvertObjectToValueForm(object obj, Expression prototype) => new(prototype.ReturnType, obj);
}