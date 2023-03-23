using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

public sealed class Dictionary : Value
{
	public Dictionary(IReadOnlyList<Type> types, Type dictionaryImplementationType) : base(dictionaryImplementationType,
		new Dictionary<Value, Value>())
	{
		if (types.Count != 2)
			throw new DictionaryMustBeInitializedWithTwoTypeParameters(dictionaryImplementationType, types);
		KeyType = types[0];
		MappedValueType = types[1];
	}

	public Type KeyType { get; }
	public Type MappedValueType { get; }

	public static Expression? TryParse(Body body, ReadOnlySpan<char> input) =>
		input.StartsWith(Base.Dictionary + '(') && input[^1] == ')'
			? new Dictionary(ParseTypeParameters(body, input), body.Method.GetType(input.ToString()))
			: null;

	private static List<Type> ParseTypeParameters(Body body, ReadOnlySpan<char> input)
	{
		var types = new List<Type>();
		foreach (var typeText in input[(Base.Dictionary.Length + 1)..^1].
			Split(',', StringSplitOptions.TrimEntries))
			types.Add(body.Method.GetType(typeText.ToString()));
		if (types.Count != 2)
			throw new DictionaryMustBeInitializedWithTwoTypeParameters(body, input.ToString());
		return types;
	}

	public sealed class DictionaryMustBeInitializedWithTwoTypeParameters : ParsingFailed
	{
		public DictionaryMustBeInitializedWithTwoTypeParameters(Type type, IReadOnlyCollection<Type> types)
			: base(type, 0,
				$"Expected Type Parameters: 2, Given type parameters: {
					types.Count
				} and they are {
					types.ToWordList()
				}") { }

		public DictionaryMustBeInitializedWithTwoTypeParameters(Body body, string expressionText) :
			base(body, expressionText) { }
	}
}