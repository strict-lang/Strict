using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Dictionary : Value
{
	public Dictionary(IReadOnlyList<Type> types, Type dictionaryImplementationType) : base(
		dictionaryImplementationType, CreateEmptyMembers(dictionaryImplementationType))
	{
		if (types.Count != 2)
			throw new DictionaryMustBeInitializedWithTwoTypeParameters(dictionaryImplementationType,
				types);
		KeyType = types[0];
		MappedValueType = types[1];
	}

	public Type KeyType { get; }
	public Type MappedValueType { get; }

	private static object CreateEmptyMembers(Type dictionaryImplementationType)
	{
		var listMemberName = dictionaryImplementationType.Members.FirstOrDefault(member =>
			member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
			member.Type.Name == Base.List)?.Name ?? Type.ElementsLowercase;
		return new System.Collections.Generic.Dictionary<string, object?>(StringComparer.Ordinal)
		{
			[listMemberName] = new System.Collections.Generic.List<object?>()
		};
	}

	public override string ToString() => ReturnType.Name;

	public static Expression? TryParse(Body body, ReadOnlySpan<char> input) =>
		input.StartsWith(Base.Dictionary + '(') && input[^1] == ')' && AreTypeParameters(body, input)
			? new Dictionary(ParseTypeParameters(body, input), body.Method.GetType(input.ToString()))
			: null;

	private static bool AreTypeParameters(Body body, ReadOnlySpan<char> input)
	{
		foreach (var typeText in input[(Base.Dictionary.Length + 1)..^1].
			Split(',', StringSplitOptions.TrimEntries))
			if (body.Method.FindType(typeText.ToString()) == null)
				return false;
		return true;
	}

	private static List<Type> ParseTypeParameters(Body body, ReadOnlySpan<char> input)
	{
		var types = new List<Type>();
		foreach (var typeText in input[(Base.Dictionary.Length + 1)..^1].
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
					types.ToWordList()
				}") { }

		public DictionaryMustBeInitializedWithTwoTypeParameters(Body body, string expressionText) :
			base(body, expressionText) { }
	}
}