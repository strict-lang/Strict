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
			: Value is IDictionary valueDictionary
				? valueDictionary.DictionaryToWordList()
				: Value is IEnumerable valueEnumerable
					? valueEnumerable.EnumerableToWordList()
					: ReturnType.IsIterator
						? $"Unknown Iterator {ReturnType.Name}: {Value}"
						: $"{ReturnType.Name}:{Value}";
}