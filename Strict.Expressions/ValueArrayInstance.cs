using System.Runtime.CompilerServices;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueArrayInstance : IEquatable<ValueArrayInstance>
{
	private static readonly ConditionalWeakTable<Type, Dictionary<string, int>> MemberIndexes = new();
	private List<ValueInstance>? items;
	private float[]? flatNumbers;
	private Type? flatElementType;
	private readonly int offset;
	private const string FlatNumbersFieldName = "flatNumbers";
	private const int MinimumCapacity = 4;
	private const int LargeGrowthChunk = 8192;

	public ValueArrayInstance(Type returnType, IEnumerable<ValueInstance> items)
	{
		ReturnType = returnType;
		if (items is IReadOnlyList<ValueInstance> readableItems && TryUseFlatNumbers(readableItems))
			return;
		this.items = new List<ValueInstance>(items);
	}

	public ValueArrayInstance(Type returnType, ValueInstance repeatedItem, int count)
	{
		ReturnType = returnType;
		if (!TryUseRepeatedFlatNumbers(repeatedItem, count))
		{
			items = new List<ValueInstance>(count);
			for (var index = 0; index < count; index++)
				items.Add(repeatedItem);
		}
	}

	private ValueArrayInstance(Type returnType, float[] flatNumbers, Type flatElementType,
		int flatElementWidth)
	{
		ReturnType = returnType;
		this.flatNumbers = flatNumbers;
		this.flatElementType = flatElementType;
		FlatWidth = flatElementWidth;
	}

	private ValueArrayInstance(Type returnType, float[] flatNumbers, Type flatElementType,
		int flatElementWidth, int offset)
	{
		ReturnType = returnType;
		this.flatNumbers = flatNumbers;
		this.flatElementType = flatElementType;
		FlatWidth = flatElementWidth;
		this.offset = offset;
	}

	/// <summary>
	/// Creates a flat numeric backing for a type where all members are numbers.
	/// No individual ValueInstances are created for each member.
	/// </summary>
	public static ValueArrayInstance CreateForTypeBacking(Type type, float[] numbers) =>
		new(type, numbers, type, numbers.Length, 0);

	public static ValueArrayInstance CreateForTypeBacking(Type type, float[] parentNumbers,
		int offset, int width) =>
		new(type, parentNumbers, type, width, offset);

	/// <summary>
	/// Returns true when all non-constant members of the type are numeric.
	/// </summary>
	public static bool IsAllNumericType(Type type)
	{
		if (type.Members.Count == 0)
			return false;
		for (var memberIndex = 0; memberIndex < type.Members.Count; memberIndex++)
			if (!type.Members[memberIndex].IsConstant && !type.Members[memberIndex].Type.IsNumber)
				return false;
		return true;
	}

	public bool TryGetMember(string name, out ValueInstance memberValue)
	{
		if (flatNumbers == null || flatElementType == null)
		{
			memberValue = default;
			return false;
		}
		var memberIndex = GetMemberIndex(name);
		if (memberIndex < 0 || memberIndex >= FlatWidth)
		{
			memberValue = default;
			return false;
		}
		memberValue = new ValueInstance(flatElementType.Members[memberIndex].Type,
			flatNumbers[offset + memberIndex]);
		return true;
	}

	public bool TrySetMember(string name, ValueInstance memberValue)
	{
		if (flatNumbers == null || flatElementType == null)
			return false;
		var memberIndex = GetMemberIndex(name);
		if (memberIndex < 0 || memberIndex >= FlatWidth)
			return false;
		flatNumbers[offset + memberIndex] = (float)memberValue.Number;
		return true;
	}

	private int GetMemberIndex(string name)
	{
		var indexes = MemberIndexes.GetValue(flatElementType!, CreateMemberIndexes);
		return indexes.TryGetValue(name, out var index)
			? index
			: -1;
	}

	private static Dictionary<string, int> CreateMemberIndexes(Type type)
	{
		var members = type.Members;
		var indexes = new Dictionary<string, int>(members.Count,
			StringComparer.OrdinalIgnoreCase);
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (!members[memberIndex].IsConstant && members[memberIndex].Type.IsNumber)
				indexes.TryAdd(members[memberIndex].Name, memberIndex);
		return indexes;
	}

	public float GetFlat(int index) => flatNumbers![offset + index];
	public void SetFlat(int index, float flatValue) => flatNumbers![offset + index] = flatValue;
	public int FlatWidth { get; private set; }

	/// <summary>
	/// Materializes a type-backing ValueArrayInstance into a ValueTypeInstance with individual
	/// member ValueInstances. Used when legacy code needs a ValueTypeInstance representation.
	/// </summary>
	public ValueTypeInstance MaterializeAsType()
	{
		var values = new ValueInstance[FlatWidth];
		for (var memberIndex = 0; memberIndex < FlatWidth; memberIndex++)
			values[memberIndex] = new ValueInstance(flatElementType!.Members[memberIndex].Type,
				flatNumbers![offset + memberIndex]);
		return new ValueTypeInstance(flatElementType!, values);
	}

	public readonly Type ReturnType;
	public List<ValueInstance> Items => items ??= MaterializeItems();
	public int Count => items?.Count ?? (flatNumbers?.Length ?? 0) / FlatWidth;

	public static ValueArrayInstance CreateWithCapacity(Type returnType, int capacity)
	{
		var instance = new ValueArrayInstance(returnType, Array.Empty<ValueInstance>());
		instance.flatNumbers = null;
		instance.flatElementType = null;
		instance.FlatWidth = 0;
		instance.items = new List<ValueInstance>(Math.Max(capacity, MinimumCapacity));
		return instance;
	}

	/// <summary>
	/// Creates a flat-backed list from a pre-built float[] without creating individual
	/// ValueInstances. Element access returns slices sharing the same backing array.
	/// </summary>
	public static ValueArrayInstance CreateFlatList(Type listType, Type elementType,
		float[] flatNumbers, int elementWidth) =>
		new(listType, flatNumbers, elementType, elementWidth, 0);

	public void Add(ValueInstance item)
	{
		if (flatNumbers != null)
			MaterializeItems();
		var currentItems = items ??= new List<ValueInstance>(MinimumCapacity);
		if (currentItems.Count == currentItems.Capacity)
			currentItems.Capacity = GetExpandedCapacity(currentItems.Capacity);
		currentItems.Add(item);
	}

	private static int GetExpandedCapacity(int currentCapacity)
	{
		if (currentCapacity < MinimumCapacity)
			return MinimumCapacity;
		return currentCapacity < LargeGrowthChunk
			? currentCapacity * 2
			: currentCapacity + LargeGrowthChunk;
	}

	public ValueInstance this[int index]
	{
		get =>
			items != null
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
			if (!TryCopyItemNumbers(sourceItems[itemIndex], elementType, elementWidth, numbers,
				itemIndex * elementWidth))
				return false;
		flatNumbers = numbers;
		flatElementType = elementType;
		FlatWidth = elementWidth;
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
		FlatWidth = elementWidth;
		return true;
	}

	private bool TryGetFlatElementLayout(out Type elementType, out int elementWidth)
	{
		if (ReturnType.IsGeneric || ReturnType is not Strict.Language.GenericTypeImplementation)
		{
			elementType = ReturnType;
			elementWidth = 0;
			return false;
		}
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
		if (item.IsFlatNumeric)
		{
			var arrayBacking = item.TryGetFlatNumericArrayInstance()!;
			for (var memberIndex = 0; memberIndex < elementWidth; memberIndex++)
				target[offset + memberIndex] = arrayBacking.GetFlat(memberIndex);
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

	public ValueArrayInstance Clone(Type newType) =>
		items != null
			? new ValueArrayInstance(newType, new List<ValueInstance>(items))
			: flatNumbers != null && flatElementType != null
				? new ValueArrayInstance(newType, (float[])flatNumbers.Clone(), flatElementType,
					FlatWidth)
				: new ValueArrayInstance(newType, []);

	private List<ValueInstance> MaterializeItems()
	{
		var createdItems = new List<ValueInstance>(Count);
		for (var itemIndex = 0; itemIndex < Count; itemIndex++)
			createdItems.Add(CreateFlatItem(itemIndex));
		flatNumbers = null;
		flatElementType = null;
		FlatWidth = 0;
		items = createdItems;
		return createdItems;
	}

	private ValueInstance CreateFlatItem(int index)
	{
		if (flatNumbers == null || flatElementType == null)
			throw new InvalidOperationException(FlatNumbersFieldName + " not initialized");
		var elementOffset = offset + index * FlatWidth;
		if (FlatWidth == 1)
			return new ValueInstance(flatElementType, flatNumbers[elementOffset]);
		if (IsAllNumericType(flatElementType))
		{
			var slice = CreateForTypeBacking(flatElementType, flatNumbers,
				elementOffset, FlatWidth);
			return new ValueInstance(slice, isFlatNumericType: true);
		}
		var values = new ValueInstance[FlatWidth];
		for (var memberIndex = 0; memberIndex < FlatWidth; memberIndex++)
			values[memberIndex] = new ValueInstance(flatElementType.Members[memberIndex].Type,
				flatNumbers[elementOffset + memberIndex]);
		return new ValueInstance(flatElementType, values);
	}

	private bool TrySetFlatItem(int index, ValueInstance value)
	{
		if (flatNumbers == null || flatElementType == null)
			return false;
		return TryCopyItemNumbers(value, flatElementType, FlatWidth, flatNumbers,
			offset + index * FlatWidth);
	}

	private static bool CanCreateRgba(Type elementType) =>
		elementType.Members.Count >= 4 &&
		elementType.Members[0].Name.Equals("Red", StringComparison.OrdinalIgnoreCase) &&
		elementType.Members[1].Name.Equals("Green", StringComparison.OrdinalIgnoreCase) &&
		elementType.Members[2].Name.Equals("Blue", StringComparison.OrdinalIgnoreCase) && elementType.
			Members[3].Name.Equals("Alpha", StringComparison.OrdinalIgnoreCase);

	public bool Equals(ValueArrayInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) && HasSameItems(other));

	private bool HasSameItems(ValueArrayInstance other)
	{
		if (Count != other.Count)
			return false;
		for (var itemIndex = 0; itemIndex < Count; itemIndex++)
			if (!this[itemIndex].Equals(other[itemIndex]))
				return false;
		return true;
	}

	public override bool Equals(object? other) => Equals(other as ValueArrayInstance);
	public override int GetHashCode() => HashCode.Combine(ReturnType, Items);
}