using System.Globalization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public readonly partial struct ValueInstance : IEquatable<ValueInstance>
{
	public int GetIteratorLength()
	{
		if (number == ListId)
			return ((ValueArrayInstance)value).Count;
		if (number == TextId)
			return ((string)value).Length;
		if (number == DictionaryId)
			throw new IteratorNotSupported(this);
		if (number == TypeId)
		{
			var typeInstance = (ValueTypeInstance)value;
			if (typeInstance.ReturnType.IsList)
				for (var i = 0; i < typeInstance.Values.Length; i++)
					if (typeInstance.Values[i].IsText)
						return typeInstance.Values[i].Text.Length; //ncrunch: no coverage
			if (typeInstance.TryGetValue("keysAndValues", out var elementsMember) &&
				elementsMember.IsList)
				return elementsMember.List.Count;
			throw new IteratorNotSupported(this);
		}
		return (int)number;
	}

	public Type GetIteratorType() => ((ValueArrayInstance)value).ReturnType.GetFirstImplementation();

	public ValueInstance GetIteratorValue(Type charTypeIfNeeded, int index)
	{
		var normalizedIndex = NormalizeIndexForIterator(index);
		return number switch
		{
			TextId => normalizedIndex >= 0 && normalizedIndex < ((string)value).Length
				? new ValueInstance(charTypeIfNeeded, ((string)value)[normalizedIndex])
				: new ValueInstance(charTypeIfNeeded, '\0'),
			ListId => ((ValueArrayInstance)value)[normalizedIndex],
			TypeId when ((ValueTypeInstance)value).ReturnType.IsList &&
				FindTextInValues((ValueTypeInstance)value, out var wrappedText) => normalizedIndex >= 0 &&
				normalizedIndex < wrappedText!.Length
					? new ValueInstance(charTypeIfNeeded, wrappedText[normalizedIndex])
					: new ValueInstance(charTypeIfNeeded, '\0'),
			TypeId when ((ValueTypeInstance)value).TryGetValue("elements", out var elementsMember) &&
				elementsMember.IsList => elementsMember.List[normalizedIndex],
			_ => throw new IteratorNotSupported(this)
		};
	}

	private int NormalizeIndexForIterator(int index) =>
		index >= 0
			? index
			: GetIteratorLength() + index;

	private static bool FindTextInValues(ValueTypeInstance typeInstance, out string? text)
	{
		for (var i = 0; i < typeInstance.Values.Length; i++)
			if (typeInstance.Values[i].IsText)
			{
				text = typeInstance.Values[i].Text;
				return true;
			} //ncrunch: no coverage start
		text = null;
		return false;
	} //ncrunch: no coverage end

	public class IteratorNotSupported(ValueInstance instance) : Exception(instance.ToString());

	public Dictionary<ValueInstance, ValueInstance> GetDictionaryItems() =>
		((ValueDictionaryInstance)value).Items;
}
