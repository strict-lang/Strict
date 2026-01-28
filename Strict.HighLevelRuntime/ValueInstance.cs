using System.Collections;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ValueInstance
{
	public ValueInstance(Type returnType, object? value)
	{
		ReturnType = returnType;
		Value = value;
		if (Value is IDictionary<string, object?> valueDictionary)
			foreach (var assignMember in valueDictionary)
				if (ReturnType.Members.All(m => m.Name != assignMember.Key))
					throw new UnableToAssignMemberToType(assignMember, valueDictionary, ReturnType);
	}

	public class UnableToAssignMemberToType(KeyValuePair<string, object?> member,
		IDictionary<string, object?> values,
		Type returnType) : ExecutionFailed(returnType,
		"Can't assign member " + member + " (of " + values.DictionaryToWordList() + ") to " +
		returnType + " " + returnType.Members.ToBrackets());

	public Type ReturnType { get; }
	public object? Value { get; }

	public override string ToString() =>
		ReturnType.Name == Base.Boolean
			? $"{Value}"
			: Value is IEnumerable valueEnumerable
				? valueEnumerable.EnumerableToWordList()
				: ReturnType.IsIterator
					? $"Unknown Iterator {ReturnType.Name}: {Value}"
					: $"{ReturnType.Name}:{Value}";
}