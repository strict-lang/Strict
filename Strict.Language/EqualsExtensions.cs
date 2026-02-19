using System.Collections;
using System.Globalization;

namespace Strict.Language;

/// <summary>
/// Comparing C# object value equality sucks we have to do a lot of work to compare two instances
/// like in out-of-the-box Strict. Used most importantly for Value expressions and ValueInstance.
/// </summary>
public static class EqualsExtensions
{
	public static bool AreEqual(object? value, object? other)
	{
		if (ReferenceEquals(value, other))
			return true;
		if (value is IList valueList && other is IList otherValueList)
			return AreListsEqual(valueList, otherValueList);
		if (value is IDictionary valueDict && other is IDictionary otherValueDict)
			return AreDictionariesEqual(valueDict, otherValueDict);
		if (IsNumeric(value) && IsNumeric(other))
			return NumberToDouble(value) == NumberToDouble(other);
		return value?.Equals(other) ?? false;
	}

	private static bool AreListsEqual(IList left, IList right)
	{
		if (left.Count != right.Count)
			return false;
		for (var i = 0; i < left.Count; i++)
			if (!AreEqual(left[i], right[i]))
				return false;
		return true;
	}

	private static bool AreDictionariesEqual(IDictionary left, IDictionary right)
	{
		if (left.Count != right.Count)
			return false;
		foreach (DictionaryEntry entry in left)
			if (!right.Contains(entry.Key) || !AreEqual(entry.Value, right[entry.Key]))
				return false;
		return true;
	}

	public static bool IsNumeric(object? value) =>
		value is sbyte or byte or short or ushort or int or uint or long or ulong or float or double
			or decimal;

	public static double NumberToDouble(object? n) => Convert.ToDouble(n, CultureInfo.InvariantCulture);
}