using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueListInstance : IEquatable<ValueListInstance>
{
 private List<ValueInstance>? items;
  private float[]? flatNumbers;
	private Type? flatElementType;
	private int flatElementWidth;
	private const string FlatNumbersFieldName = "flatNumbers";
	public ValueListInstance(Type returnType, IEnumerable<ValueInstance> items)
	{
		ReturnType = returnType;
   if (items is IReadOnlyList<ValueInstance> readableItems && TryUseFlatNumbers(readableItems))
			return;
		this.items = new List<ValueInstance>(items);
	}

	public ValueListInstance(Type returnType, List<ValueInstance> items)
	{
		ReturnType = returnType;
    if (!TryUseFlatNumbers(items))
			this.items = items;
	}

	public ValueListInstance(Type returnType, ValueInstance repeatedItem, int count)
	{
		ReturnType = returnType;
		if (!TryUseRepeatedFlatNumbers(repeatedItem, count))
		{
			items = new List<ValueInstance>(count);
			for (var index = 0; index < count; index++)
				items.Add(repeatedItem);
		}
	}

	private ValueListInstance(Type returnType, float[] flatNumbers, Type flatElementType,
		int flatElementWidth)
	{
		ReturnType = returnType;
		this.flatNumbers = flatNumbers;
		this.flatElementType = flatElementType;
		this.flatElementWidth = flatElementWidth;
	}

	public readonly Type ReturnType;
  public List<ValueInstance> Items => items ??= MaterializeItems();
	public int Count => items?.Count ?? (flatNumbers?.Length ?? 0) / flatElementWidth;
	public ValueInstance this[int index]
	{
		get => items != null
			? items[index]
			: CreateFlatItem(index);
		set
		{
			if (items != null)
			{
				items[index] = value;
				return;
			}
			if (!TrySetFlatItem(index, value))
				MaterializeItems()[index] = value;
		}
	}

	private bool TryUseFlatNumbers(IReadOnlyList<ValueInstance> sourceItems)
	{
		if (!TryGetFlatElementLayout(out var elementType, out var elementWidth))
			return false;
   var numbers = new float[sourceItems.Count * elementWidth];
		for (var itemIndex = 0; itemIndex < sourceItems.Count; itemIndex++)
			if (!TryCopyItemNumbers(sourceItems[itemIndex], elementType, elementWidth,
				numbers, itemIndex * elementWidth))
				return false;
		flatNumbers = numbers;
		flatElementType = elementType;
		flatElementWidth = elementWidth;
		return true;
	}

	private bool TryUseRepeatedFlatNumbers(ValueInstance repeatedItem, int count)
	{
		if (!TryGetFlatElementLayout(out var elementType, out var elementWidth))
			return false;
   var numbers = new float[count * elementWidth];
		if (!TryCopyItemNumbers(repeatedItem, elementType, elementWidth, numbers, 0))
			return false;
		for (var itemIndex = 1; itemIndex < count; itemIndex++)
			Array.Copy(numbers, 0, numbers, itemIndex * elementWidth, elementWidth);
		flatNumbers = numbers;
		flatElementType = elementType;
		flatElementWidth = elementWidth;
		return true;
	}

	private bool TryGetFlatElementLayout(out Type elementType, out int elementWidth)
	{
		elementType = ReturnType.GetFirstImplementation();
		if (elementType.IsNumber)
		{
			elementWidth = 1;
			return true;
		}
		if (!elementType.IsDataType || elementType.Members.Count == 0)
		{
			elementWidth = 0;
			return false;
		}
		for (var memberIndex = 0; memberIndex < elementType.Members.Count; memberIndex++)
			if (!elementType.Members[memberIndex].Type.IsNumber)
			{
				elementWidth = 0;
				return false;
			}
		elementWidth = elementType.Members.Count;
		return true;
	}

  private static bool TryCopyItemNumbers(ValueInstance item, Type elementType, int elementWidth,
		float[] target, int offset)
	{
		if (elementWidth == 1)
		{
     target[offset] = (float)item.Number;
			return true;
		}
		if (item.TryGetPackedRgbaChannels(out var red, out var green, out var blue, out var alpha) &&
			elementWidth == 4)
		{
     target[offset] = (float)red;
			target[offset + 1] = (float)green;
			target[offset + 2] = (float)blue;
			target[offset + 3] = (float)alpha;
			return true;
		}
		var typeInstance = item.TryGetValueTypeInstance();
		if (typeInstance == null || !typeInstance.ReturnType.IsSameOrCanBeUsedAs(elementType) ||
			typeInstance.Values.Length < elementWidth)
			return false;
		for (var memberIndex = 0; memberIndex < elementWidth; memberIndex++)
     target[offset + memberIndex] = (float)typeInstance.Values[memberIndex].Number;
		return true;
	}

	public ValueListInstance Clone() =>
		items != null
			? new ValueListInstance(ReturnType, new List<ValueInstance>(items))
			: flatNumbers != null && flatElementType != null
				? new ValueListInstance(ReturnType, (float[])flatNumbers.Clone(), flatElementType,
					flatElementWidth)
				: new ValueListInstance(ReturnType, []);

	private List<ValueInstance> MaterializeItems()
	{
		var createdItems = new List<ValueInstance>(Count);
		for (var itemIndex = 0; itemIndex < Count; itemIndex++)
			createdItems.Add(CreateFlatItem(itemIndex));
		flatNumbers = null;
		flatElementType = null;
		flatElementWidth = 0;
		return createdItems;
	}

	private ValueInstance CreateFlatItem(int index)
	{
		if (flatNumbers == null || flatElementType == null)
			throw new InvalidOperationException(FlatNumbersFieldName + " not initialized");
		var offset = index * flatElementWidth;
		if (flatElementWidth == 1)
			return new ValueInstance(flatElementType, flatNumbers[offset]);
		if (flatElementWidth == 4 && CanCreateRgba(flatElementType))
			return ValueInstance.CreateRgba(flatElementType, flatNumbers[offset],
				flatNumbers[offset + 1], flatNumbers[offset + 2], flatNumbers[offset + 3]);
		var values = new ValueInstance[flatElementWidth];
		for (var memberIndex = 0; memberIndex < flatElementWidth; memberIndex++)
			values[memberIndex] = new ValueInstance(flatElementType.Members[memberIndex].Type,
				flatNumbers[offset + memberIndex]);
		return new ValueInstance(flatElementType, values);
	}

	private bool TrySetFlatItem(int index, ValueInstance value)
	{
		if (flatNumbers == null || flatElementType == null)
			return false;
		return TryCopyItemNumbers(value, flatElementType, flatElementWidth, flatNumbers,
			index * flatElementWidth);
	}

	private static bool CanCreateRgba(Type elementType) =>
		elementType.Members.Count >= 4 &&
		elementType.Members[0].Name.Equals("Red", StringComparison.OrdinalIgnoreCase) &&
		elementType.Members[1].Name.Equals("Green", StringComparison.OrdinalIgnoreCase) &&
		elementType.Members[2].Name.Equals("Blue", StringComparison.OrdinalIgnoreCase) &&
		elementType.Members[3].Name.Equals("Alpha", StringComparison.OrdinalIgnoreCase);

	public bool Equals(ValueListInstance? other) =>
   other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) && HasSameItems(other));

	private bool HasSameItems(ValueListInstance other)
	{
		if (Count != other.Count)
			return false;
		for (var itemIndex = 0; itemIndex < Count; itemIndex++)
			if (!this[itemIndex].Equals(other[itemIndex]))
				return false;
		return true;
	}

	//ncrunch: no coverage start
	public override bool Equals(object? other) => Equals(other as ValueListInstance);
	public override int GetHashCode() => HashCode.Combine(ReturnType, Items);
}