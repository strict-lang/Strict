using System.Globalization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Optimized for size, always just contains 2 values, a pointer to the type, string, list,
/// dictionary, or type instance and if it is a primitive (most common, most lines just return
/// None or True) the None, Boolean, or Number data.
/// </summary>
public readonly partial struct ValueInstance : IEquatable<ValueInstance>
{
	/*TODO: nah
	//TODO: stupid, remove!
	private sealed class PackedRgbaType(Type returnType)
	{
		public readonly Type ReturnType = returnType;
	}

	//TODO: stupid, remove!
	private static readonly ConditionalWeakTable<Type, PackedRgbaType> PackedRgbaTypes = new();
	private bool IsPackedRgba => value is PackedRgbaType;
	private PackedRgbaType RgbaType => (PackedRgbaType)value;

	//TODO: stupid, remove!
	private static PackedRgbaType GetPackedRgbaType(Type returnType) =>
		PackedRgbaTypes.GetValue(returnType, type => new PackedRgbaType(type));

	public static ValueInstance CreateRgba(Type returnType, double red, double green, double blue,
		double alpha) =>
		new(GetPackedRgbaType(returnType), PackRgba(red, green, blue, alpha));

	private ValueInstance(PackedRgbaType packedType, double packedNumber)
	{
		TrackCreation();
		value = packedType;
		number = packedNumber;
	}

	//TODO: stupid, remove!
	private static double PackRgba(double red, double green, double blue, double alpha)
	{
		var packed = (ushort)BitConverter.HalfToInt16Bits((Half)red) |
			(ulong)(ushort)BitConverter.HalfToInt16Bits((Half)green) << 16 |
			(ulong)(ushort)BitConverter.HalfToInt16Bits((Half)blue) << 32 |
			(ulong)(ushort)BitConverter.HalfToInt16Bits((Half)alpha) << 48;
		return BitConverter.Int64BitsToDouble((long)packed);
	}

	//TODO: stupid, remove!
	private static double UnpackRgba(double packedNumber, int shift) =>
		(float)BitConverter.Int16BitsToHalf(
			(short)((ulong)BitConverter.DoubleToInt64Bits(packedNumber) >> shift));

	//TODO: stupid, remove!
	private ValueInstance CreatePackedRgbaComponent(int memberIndex) =>
		new(RgbaType.ReturnType.Members[memberIndex].Type, UnpackRgba(number, memberIndex * 16));

	//TODO: stupid, remove!
	private static bool IsRgbaMemberName(IReadOnlyList<Strict.Language.Member> members, int index,
		string expectedName) =>
		index < members.Count && members[index].Name.Equals(expectedName,
			StringComparison.OrdinalIgnoreCase);

	//TODO: stupid, remove!
	private static bool CanUsePackedRgba(Type returnType, IReadOnlyList<ValueInstance> values)
	{
		var members = returnType.Members;
		return values.Count == 4 && members.Count >= 4 && IsRgbaMemberName(members, 0, "Red") &&
			IsRgbaMemberName(members, 1, "Green") && IsRgbaMemberName(members, 2, "Blue") &&
			IsRgbaMemberName(members, 3, "Alpha") && values[0].GetType().IsNumber &&
			values[1].GetType().IsNumber && values[2].GetType().IsNumber &&
			values[3].GetType().IsNumber;
	}

	//TODO: stupid, remove!
	private ValueTypeInstance MaterializePackedRgba() =>
		new(RgbaType.ReturnType, [
			CreatePackedRgbaComponent(0),
			CreatePackedRgbaComponent(1),
			CreatePackedRgbaComponent(2),
			CreatePackedRgbaComponent(3)
		]);

	//TODO: stupid, remove!
	public bool TryGetPackedRgbaMember(string memberName, out ValueInstance component)
	{
		if (!IsPackedRgba)
		{
			component = default;
			return false;
		}
		component = memberName switch
		{
			"Red" => CreatePackedRgbaComponent(0),
			"Green" => CreatePackedRgbaComponent(1),
			"Blue" => CreatePackedRgbaComponent(2),
			"Alpha" => CreatePackedRgbaComponent(3),
			_ => default
		};
		return component.HasValue;
	}

	//TODO: stupid, remove!
	public bool TryGetPackedRgbaChannels(out double red, out double green, out double blue,
		out double alpha)
	{
		if (!IsPackedRgba)
		{
			red = green = blue = alpha = 0;
			return false;
		}
		red = UnpackRgba(number, 0);
		green = UnpackRgba(number, 16);
		blue = UnpackRgba(number, 32);
		alpha = UnpackRgba(number, 48);
		return true;
	}

	//TODO: stupid, remove!
	private string ToPackedRgbaText()
	{
		var red = CreatePackedRgbaComponent(0).ToExpressionCodeString();
		var green = CreatePackedRgbaComponent(1).ToExpressionCodeString();
		var blue = CreatePackedRgbaComponent(2).ToExpressionCodeString();
		var alpha = CreatePackedRgbaComponent(3).Number;
		return alpha == 1
			? "(" + red + ", " + green + ", " + blue + ")"
			: "(" + red + ", " + green + ", " + blue + ", " +
			CreatePackedRgbaComponent(3).ToExpressionCodeString() + ")";
	}
*/

	public ValueInstance(Type noneReturnType)
	{
		TrackCreation();
		value = noneReturnType;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(Type=" + noneReturnType + ")");
#endif
	}

	public static int CreatedCount;

	private static int creationLimit = int.MaxValue;

	public static void SetCreationLimit(int newLimit)
	{
		CreatedCount = 0;
		creationLimit = newLimit;
	}

	public sealed class CreationLimitExceeded(long created, long limit)
		: Exception("ValueInstance creation limit exceeded: " + created + " > " + limit);

	private static void TrackCreation()
	{
		CreatedCount++;
		if (CreatedCount > creationLimit)
			throw new CreationLimitExceeded(CreatedCount, creationLimit);
	}

	/// <summary>
	/// If number is TextId, value points to a string (only non-Mutable, for Mutable TypeId is used)
	/// If number is ListId, value points to ValueArrayInstance (ReturnType and Items).
	/// If number is DictionaryId, value points to ValueDictionaryInstance (ReturnType and Items).
	/// If number is TypeId, then this points to a TypeValueInstance containing with ReturnType.
	/// In all other cases this is a primitive (None, Boolean, Number), and value is the ReturnType
	/// </summary>
	private readonly object value;

	/// <summary>
	/// Stores the value only if it is a None, Boolean, or Number. Otherwise use value below.
	/// </summary>
	internal readonly double number;

	/// <summary>
	/// These are all unsupported double values, which we don't allow or support and use to id types
	/// </summary>
	private const double TextId = -7.90897526e307;

	private const double ListId = -7.81590825e307;

	private const double DictionaryId = -7.719027815e307;

	private const double TypeId = -7.657178621e307;

	private const double FlatNumericId = -7.595329427e307;

	public ValueInstance(Type booleanReturnType, bool isTrue)
	{
		TrackCreation();
		value = booleanReturnType;
		number = isTrue
			? 1
			: 0;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(Type=" + booleanReturnType + ", number=" + isTrue + ")");
#endif
	}

	/// <summary>
	/// Creates a flat numeric type where all members are backed by a float[] without
	/// creating individual ValueInstances for each member. Use this instead of
	/// new ValueInstance(type, ValueInstance[]) when the type is all-numeric.
	/// </summary>
	public static ValueInstance CreateFlatNumericType(Type type, float[] numbers)
	{
		var backing = ValueArrayInstance.CreateForTypeBacking(type, numbers);
		return new ValueInstance(backing, true);
	}

	internal ValueInstance(ValueArrayInstance backing, bool isFlatNumericType)
	{
		TrackCreation();
		value = backing;
		number = isFlatNumericType
			? FlatNumericId
			: ListId;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(" + (isFlatNumericType
				? "FlatNumericType"
				: "List") + "=" + backing.ReturnType + ")");
#endif
	}

	public ValueInstance(Type numberReturnType, double setNumber)
	{
		if (setNumber is TextId or ListId or DictionaryId or TypeId or FlatNumericId)
			throw new InvalidTypeValue(numberReturnType, setNumber);
		value = numberReturnType;
		number = setNumber;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(Type=" + numberReturnType + ", number=" + setNumber + ")");
#endif
	}

	public sealed class InvalidTypeValue(Type returnType, object value) : ParsingFailed(returnType,
		0, BuildInvalidTypeValueMessage(returnType, value));

	public ValueInstance(string text)
	{
		TrackCreation();
		value = text;
		number = TextId;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(text=" + text + ")");
#endif
	}

	public ValueInstance(Type returnType, Dictionary<ValueInstance, ValueInstance> dictionary)
	{
		TrackCreation();
		value = new ValueDictionaryInstance(returnType, dictionary);
		number = DictionaryId;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(Type=" + returnType + ", dictionaryCount=" + dictionary.Count + ")");
#endif
	}

	public ValueInstance(Type returnType, ValueInstance[] values)
	{
		TrackCreation();
		if (returnType.IsList)
		{
			value = new ValueArrayInstance(returnType, values);
			number = ListId;
#if DEBUG
			if (PerformanceLog.IsEnabled)
				LogCreated("ctor(Type=" + returnType + ", values=" + DescribeValues(values) + ")");
#endif
			return;
		}
		if (!returnType.IsMutable && (returnType.IsNumber || returnType.IsText ||
			returnType.IsCharacter || returnType.IsDictionary || returnType.IsEnum ||
			returnType.IsBoolean || returnType.IsNone))
			throw new ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes(returnType);
		/*obs, this is not the way!
		if (CanUsePackedRgba(returnType, values))
		{
			value = GetPackedRgbaType(returnType);
			number = PackRgba(values[0].Number, values[1].Number, values[2].Number, values[3].Number);
			if (PerformanceLog.IsEnabled)
				LogCreated("ctor(Type=" + returnType + ", values=" + DescribeValues(values) + ")");
			return;
		}
		*/
		if (ValueArrayInstance.IsAllNumericType(returnType))
		{
			var flatNumbers = CreateFlatNumbers(returnType, values);
			for (var flatIndex = 0; flatIndex < values.Length; flatIndex++)
				if (values[flatIndex].HasValue)
					flatNumbers[flatIndex] = (float)values[flatIndex].GetArithmeticNumber();
			value = ValueArrayInstance.CreateForTypeBacking(returnType, flatNumbers);
			number = FlatNumericId;
#if DEBUG
			if (PerformanceLog.IsEnabled)
				LogCreated("ctor(FlatNumeric=" + returnType + ", values=" + DescribeValues(values) + ")");
#endif
			return;
		}
		value = new ValueTypeInstance(returnType, values);
		number = TypeId;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(Type=" + returnType + ", values=" + DescribeValues(values) + ")");
#endif
	}

	private static float[] CreateFlatNumbers(Type returnType, IReadOnlyList<ValueInstance> values)
	{
		var members = returnType.Members;
		var flatNumbers = new float[Math.Max(values.Count, members.Count)];
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
		{
			if (memberIndex < values.Count && values[memberIndex].HasValue)
				continue;
			if (members[memberIndex].InitialValue is Value initialValue)
				flatNumbers[memberIndex] = (float)initialValue.Data.Number;
			else if (members[memberIndex].InitialValue is MethodCall
				{
					Method.Name: Method.From
				} methodCall && methodCall.Arguments.Count > 0 &&
				methodCall.Arguments[0] is Value methodCallValue)
				flatNumbers[memberIndex] = (float)methodCallValue.Data.Number;
			else if (members[memberIndex].InitialValue != null)
				flatNumbers[memberIndex] = 1;
		}
		return flatNumbers;
	}

	/*TODO, this is not the way this should be called!
	/// <summary>
	/// Creates an empty list instance with preallocated capacity to reduce growth allocations.
	/// </summary>
	public static ValueInstance CreateListWithCapacity(Type returnType, int listCapacity) =>
		returnType.IsList
			? new ValueInstance(ValueArrayInstance.CreateWithCapacity(returnType, listCapacity))
			: throw new ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes(returnType);

	/// <summary>
	/// Creates a flat numeric list from a pre-built float[] without creating individual
	/// ValueInstances for each element. Element access returns slices of the same backing array.
	/// </summary>
	public static ValueInstance CreateFlatNumericList(Type listType, Type elementType,
		float[] flatNumbers, int elementWidth)
	{
		var backing = ValueArrayInstance.CreateFlatList(listType, elementType, flatNumbers,
			elementWidth);
		return new ValueInstance(backing);
	}
*/

	public ValueInstance(Type returnType, ValueInstance repeatedItem, int count)
	{
		TrackCreation();
		if (!returnType.IsList)
			throw new ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes(returnType);
		value = new ValueArrayInstance(returnType, repeatedItem, count);
		number = ListId;
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(Type=" + returnType + ", repeatCount=" + count + ")");
#endif
	}

	public ValueInstance(ValueArrayInstance listInstance)
	{
		TrackCreation();
		value = listInstance;
		number = ListId;
	}

	public class ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes(Type returnType)
		: Exception(returnType.ToString()) { }

	/// <summary>
	/// Used by ApplyMethodReturnTypeMutable and TryEvaluate to flip if this is a mutable or not.
	/// </summary>
	public ValueInstance(ValueInstance existingInstance, Type newType)
	{
		TrackCreation();
		/*TODO: this is not the way
		if (existingInstance.IsPackedRgba)
		{
			if (CanUsePackedRgba(newType, [
					existingInstance.CreatePackedRgbaComponent(0),
					existingInstance.CreatePackedRgbaComponent(1),
					existingInstance.CreatePackedRgbaComponent(2),
					existingInstance.CreatePackedRgbaComponent(3)
				]))
			{
				value = GetPackedRgbaType(newType);
				number = existingInstance.number;
			}
			else
			{
				value = new ValueTypeInstance(newType, [
					existingInstance.CreatePackedRgbaComponent(0),
					existingInstance.CreatePackedRgbaComponent(1),
					existingInstance.CreatePackedRgbaComponent(2),
					existingInstance.CreatePackedRgbaComponent(3)
				]);
				number = TypeId;
			}
			if (PerformanceLog.IsEnabled)
				LogCreated("ctor(existingInstance=" + DescribeValue(existingInstance) + ", newType=" +
					newType + ")");
			return;
		}
		*/
		switch (existingInstance.number)
		{
		case TextId:
			var textTypeMembers = newType.Members;
			var textValues = new ValueInstance[textTypeMembers.Count];
			var textValue = new ValueInstance((string)existingInstance.value);
			if (newType.IsList)
			{
				var elementsMemberIndex = 0;
				for (var memberIndex = 0; memberIndex < textTypeMembers.Count; memberIndex++)
					if (textTypeMembers[memberIndex].Name == Type.ElementsLowercase)
					{
						elementsMemberIndex = memberIndex;
						break;
					}
				textValues[elementsMemberIndex] = textValue;
			}
			else
			{
				textValues[0] = textValue;
			}
			value = new ValueTypeInstance(newType, textValues);
			number = TypeId;
			break;
		case ListId:
			value = ((ValueArrayInstance)existingInstance.value).Clone(newType);
			number = ListId;
			break;
		case DictionaryId:
			value = new ValueDictionaryInstance(newType,
				((ValueDictionaryInstance)existingInstance.value).Items);
			number = DictionaryId;
			break;
		case TypeId:
			var existingTypeInstance = (ValueTypeInstance)existingInstance.value;
			if (!newType.IsMutable && existingTypeInstance.ReturnType.IsMutable && newType.IsText)
			{
				value = existingTypeInstance.Values[0].value;
				number = TextId;
			}
			else
			{
				value = new ValueTypeInstance(newType, existingTypeInstance.Values);
				number = TypeId;
			}
			break;
		case FlatNumericId:
			var existingFlatArray = (ValueArrayInstance)existingInstance.value;
			if (ValueArrayInstance.IsAllNumericType(newType))
			{
				var clonedNumbers = new float[existingFlatArray.FlatWidth];
				for (var flatIndex = 0; flatIndex < existingFlatArray.FlatWidth; flatIndex++)
					clonedNumbers[flatIndex] = existingFlatArray.GetFlat(flatIndex);
				value = ValueArrayInstance.CreateForTypeBacking(newType, clonedNumbers);
				number = FlatNumericId;
			}
			else
			{
				value = new ValueTypeInstance(newType, existingFlatArray.MaterializeAsType().Values);
				number = TypeId;
			}
			break;
		default:
			value = newType;
			number = existingInstance.number;
			break;
		}
#if DEBUG
		if (PerformanceLog.IsEnabled)
			LogCreated("ctor(existingInstance=" + DescribeValue(existingInstance) + ", newType=" +
				newType + ")");
#endif
	}

	public bool IsType(Type type) =>
		type is OneOfType oneOf
			? oneOf.Types.Any(IsType)
			: /*TODO: IsPackedRgba
				? type == RgbaType.ReturnType
				:*/ number switch
			{
				TextId => type.IsText,
				ListId => type == ((ValueArrayInstance)value).ReturnType,
				DictionaryId => type == ((ValueDictionaryInstance)value).ReturnType,
				TypeId => type == ((ValueTypeInstance)value).ReturnType,
				FlatNumericId => type == ((ValueArrayInstance)value).ReturnType,
				_ => IsPrimitiveType(type)
			};

	public bool IsPrimitiveType(Type noneBoolOrNumberType) => value == noneBoolOrNumberType;

	public bool HasValue => value != null;

	public bool IsText => number is TextId;

	public string Text => (string)value;

	public double Number => number;

	public bool Boolean => number != 0;

	public bool IsList => number is ListId;

	public ValueArrayInstance List => (ValueArrayInstance)value;

	public bool IsDictionary => number is DictionaryId;

	public bool IsNumberLike(Type numberType) =>
		IsPrimitiveType(numberType) || (!IsText && !IsList && !IsDictionary &&
			GetType().IsSameOrCanBeUsedAs(numberType));

	public double GetArithmeticNumber()
	{
		/*TODO
		if (IsPackedRgba)
			return UnpackRgba(number, 0);
			*/
		if (number == FlatNumericId)
			return ((ValueArrayInstance)value).GetFlat(0);
		if (number != TypeId)
			return number;
		var typeInstance = (ValueTypeInstance)value;
		return typeInstance.Values.Length > 0
			? typeInstance.Values[0].number
			: 0;
	}

	public bool IsSameOrCanBeUsedAs(Type otherType) =>
		/*TODO
		IsPackedRgba
			? RgbaType.ReturnType.IsSameOrCanBeUsedAs(otherType)
			: */number switch
		{
			TextId => otherType.IsText || (otherType.IsList && otherType is GenericTypeImplementation
			{
				ImplementationTypes: [{ IsCharacter: true }]
			}),
			ListId => ((ValueArrayInstance)value).ReturnType.IsSameOrCanBeUsedAs(otherType),
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType.IsSameOrCanBeUsedAs(otherType),
			TypeId => ((ValueTypeInstance)value).ReturnType.IsSameOrCanBeUsedAs(otherType),
			FlatNumericId => ((ValueArrayInstance)value).ReturnType.IsSameOrCanBeUsedAs(otherType),
			_ => ((Type)value).IsSameOrCanBeUsedAs(otherType)
		};

	public bool IsValueTypeInstanceType =>
		number == TypeId && value is ValueTypeInstance { ReturnType.Name: nameof(Type) };

	public ValueTypeInstance? TryGetValueTypeInstance() =>
		/*TODO: this is not the way, should be general!
		IsPackedRgba
			? MaterializePackedRgba()
			: */number switch
		{
			TypeId => (ValueTypeInstance)value,
			FlatNumericId => ((ValueArrayInstance)value).MaterializeAsType(),
			_ => null
		};

	public bool IsFlatNumeric => number is FlatNumericId;

	public ValueArrayInstance? TryGetFlatNumericArrayInstance() =>
		number == FlatNumericId
			? (ValueArrayInstance)value
			: null;

	public bool TryGetFlatNumericMember(string memberName, out ValueInstance memberValue)
	{
		if (number == FlatNumericId)
			return ((ValueArrayInstance)value).TryGetMember(memberName, out memberValue);
		memberValue = default;
		return false;
	}

	/// <summary>
	/// Special code to make the ValueInstance mutable if the method return type requires it (rare)
	/// </summary>
	public ValueInstance ApplyMethodReturnTypeMutable(Type methodReturnType)
	{
		if (IsList && methodReturnType.IsList && GetType() != methodReturnType &&
			IsSameOrCanBeUsedAs(methodReturnType))
			return new ValueInstance(this, methodReturnType);
		var isInstanceMutable = IsMutable;
		if (isInstanceMutable == methodReturnType.IsMutable)
			return this;
		if (!isInstanceMutable)
			return IsSameOrCanBeUsedAs(methodReturnType.GetFirstImplementation())
				? new ValueInstance(this, methodReturnType)
				: this;
		return GetType().GetFirstImplementation().IsSameOrCanBeUsedAs(methodReturnType)
			? new ValueInstance(this, methodReturnType)
			: this;
	}

	/// <summary>
	/// Gets the underlying type, except if it is a Text, that should always be checked before.
	/// </summary>
	public new Type GetType() =>
		/*TODO: IsPackedRgba
			? RgbaType.ReturnType
			: */number switch
		{
			ListId => ((ValueArrayInstance)value).ReturnType,
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType,
			TypeId => ((ValueTypeInstance)value).ReturnType,
			FlatNumericId => ((ValueArrayInstance)value).ReturnType,
			_ => (Type)value
		};

	public bool IsMutable =>
		/*TODO
		IsPackedRgba
			? RgbaType.ReturnType.IsMutable
			: */number switch
		{
			TextId => false,
			ListId => ((ValueArrayInstance)value).ReturnType.IsMutable,
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType.IsMutable,
			TypeId => ((ValueTypeInstance)value).ReturnType.IsMutable,
			FlatNumericId => ((ValueArrayInstance)value).ReturnType.IsMutable,
			_ => ((Type)value).IsMutable
		};

	public bool IsError =>
		/*TODO: IsPackedRgba
			? RgbaType.ReturnType.IsError
			: */(number is TypeId && ((ValueTypeInstance)value).ReturnType.IsError) ||
		(number is FlatNumericId && ((ValueArrayInstance)value).ReturnType.IsError);

	/// <summary>
	/// Equals is a bit more complex as we need to handle all different kinds of ValueInstances we
	/// could have, either primary, text, list, dictionary, or type. Or any of those mixed.
	/// </summary>
	public bool Equals(ValueInstance other)
	{
		//TODO: this is way too complicated!
		if (number == other.number && value == other.value)
			return true;
		/*TODO
		if (IsPackedRgba)
		{
			var otherTypeInstance = other.TryGetValueTypeInstance();
			if (otherTypeInstance == null ||
				!otherTypeInstance.ReturnType.IsSameOrCanBeUsedAs(RgbaType.ReturnType))
				return false;
			return CreatePackedRgbaComponent(0).Equals(otherTypeInstance.Values[0]) &&
				CreatePackedRgbaComponent(1).Equals(otherTypeInstance.Values[1]) &&
				CreatePackedRgbaComponent(2).Equals(otherTypeInstance.Values[2]) &&
				CreatePackedRgbaComponent(3).Equals(otherTypeInstance.Values[3]);
		}
		if (other.IsPackedRgba)
			return other.Equals(this);
			*/
		ComplexEqualsCalls++;
		if (number == FlatNumericId)
		{
			var flatArray = (ValueArrayInstance)value;
			if (other.number == FlatNumericId)
			{
				var otherFlatArray = (ValueArrayInstance)other.value;
				if (!flatArray.ReturnType.IsSameOrCanBeUsedAs(otherFlatArray.ReturnType) ||
					flatArray.FlatWidth != otherFlatArray.FlatWidth)
					return false;
				for (var flatIndex = 0; flatIndex < flatArray.FlatWidth; flatIndex++)
					if (!AreSameFlatNumbers(flatArray.ReturnType, flatArray.GetFlat(flatIndex),
						otherFlatArray.GetFlat(flatIndex)))
						return false;
				return true;
			}
			if (other.number != ListId && other.number != DictionaryId && other.number != TextId &&
				other.value is not Type { IsNone: true } &&
				flatArray.TryGetMember("number", out var flatNumberMember) &&
				flatNumberMember.number == other.number)
				return true;
			return flatArray.MaterializeAsType().Equals(other.TryGetValueTypeInstance());
		}
		if (other.number == FlatNumericId)
			return other.Equals(this);
		if (number == TypeId)
		{
			var instance = (ValueTypeInstance)value;
			if (other.number == TypeId)
				return instance.Equals((ValueTypeInstance)other.value);
			if (other.number != ListId && other.number != DictionaryId && other.number != TextId &&
				other.value is not Type { IsNone: true } &&
				instance.TryGetValue("number", out var numberMember) && numberMember.number == other.number)
				return true;
		}
		else if (other.number == TypeId && number != ListId && number != DictionaryId &&
			number != TextId && value is not Type { IsNone: true } &&
			((ValueTypeInstance)other.value).TryGetValue("number", out var otherNumberMember) &&
			otherNumberMember.number == number)
		{
			return true;
		}
		if (number != other.number)
			return false;
		if (number == TextId)
			return (string)value == (string)other.value;
		if (number == ListId)
			return ((ValueArrayInstance)value).Equals((ValueArrayInstance)other.value);
		return number == DictionaryId
			? ((ValueDictionaryInstance)value).Equals((ValueDictionaryInstance)other.value)
			: ((Type)other.value).IsSameOrCanBeUsedAs((Type)value);
	}

	private static bool AreSameFlatNumbers(Type returnType, float number, float otherNumber) =>
		number == otherNumber || (returnType.Name == "ColorValue" &&
			Math.Abs(number - otherNumber) <= 1.0 / 255.0);

	public static int ComplexEqualsCalls;

	public override int GetHashCode() => HashCode.Combine(number, value);
}
