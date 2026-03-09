using Strict.Expressions;

namespace Strict.Runtime;

/// <summary>
/// Fixed-size array-backed register file for the 16 <see cref="Register"/> slots.
/// Replaces <c>Dictionary&lt;Register, ValueInstance&gt;</c>: array indexing is O(1) with no
/// hash overhead and a single allocation instead of the dictionary's internal bucket arrays.
/// </summary>
public sealed class RegisterFile
{
	private readonly ValueInstance[] data = new ValueInstance[16];
	public ValueInstance this[Register r]
	{
		get => data[(int)r];
		set => data[(int)r] = value;
	}

	/// <summary>
	/// Returns false (and a default value) only when the slot has never been written.
	/// </summary>
	internal bool TryGet(Register r, out ValueInstance value)
	{
		value = data[(int)r];
		return !EqualityComparer<ValueInstance>.Default.Equals(value, default);
	}

	public void Clear() => Array.Clear(data, 0, 16);
}