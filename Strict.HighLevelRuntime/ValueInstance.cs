using System.Collections;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ValueInstance(Type returnType, object? value)
{
	public Type ReturnType { get; } = returnType;
	public object? Value { get; } = value;

	public override string ToString() =>
		ReturnType.Name == Base.Boolean
			? $"{Value}"
			: ReturnType.IsIterator
				? (Value is IDictionary valueDictionary
					? valueDictionary.DictionaryToWordList()
					: Value is IEnumerable valueEnumerable
						? valueEnumerable.EnumerableToWordList()
						: $"Unknown Iterator: {Value}")
				: $"{ReturnType.Name}:{Value}";
}