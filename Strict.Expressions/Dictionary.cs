using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Dictionary : Value
{
	public Dictionary(IReadOnlyList<Type> types, Type dictionaryImplementationType) : base(
		dictionaryImplementationType, new ValueInstance(dictionaryImplementationType,
			new Dictionary<ValueInstance, ValueInstance>()))
	{
		if (types.Count != 2)
			throw new DictionaryMustBeInitializedWithTwoTypeParameters(dictionaryImplementationType,
				types);
		KeyType = types[0];
		MappedValueType = types[1];
	}

	public Type KeyType { get; }
	public Type MappedValueType { get; }
	public override string ToString() => ReturnType.Name;

	public static Expression? TryParse(Body body, ReadOnlySpan<char> input) =>
		input.StartsWith(Type.Dictionary + '(') && input[^1] == ')' && AreTypeParameters(body, input)
			? new Dictionary(ParseTypeParameters(body, input), body.Method.GetType(input.ToString()))
			: null;

	private static bool AreTypeParameters(Body body, ReadOnlySpan<char> input)
	{
		foreach (var typeText in input[(Type.Dictionary.Length + 1)..^1].
			Split(',', StringSplitOptions.TrimEntries))
			if (body.Method.FindType(typeText.ToString()) == null)
				return false;
		return true;
	}

	private static List<Type> ParseTypeParameters(Body body, ReadOnlySpan<char> input)
	{
		var types = new List<Type>();
		foreach (var typeText in input[(Type.Dictionary.Length + 1)..^1].
			Split(',', StringSplitOptions.TrimEntries))
			types.Add(body.Method.GetType(typeText.ToString()));
		return types.Count != 2
			? throw new DictionaryMustBeInitializedWithTwoTypeParameters(body, input.ToString())
			: types;
	}

	public sealed class DictionaryMustBeInitializedWithTwoTypeParameters : ParsingFailed
	{
		public DictionaryMustBeInitializedWithTwoTypeParameters(Type type, IReadOnlyCollection<Type> types)
			: base(type, 0,
				$"Expected Type Parameters: 2, Given type parameters: {
					types.Count
				} and they are {
					string.Join(", ", types)
				}") { }

		public DictionaryMustBeInitializedWithTwoTypeParameters(Body body, string expressionText) :
			base(body, expressionText) { }
	}
}