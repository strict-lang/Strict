using Strict.Bytecode;
using Strict.Expressions;

namespace Strict;

/// <summary>
/// Fast fixed-size array-backed register file for the 16 <see cref="Register"/> slots.
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
		return value.HasValue;
	}

	public void SaveTo(ValueInstance[] snapshot) => Array.Copy(data, snapshot, 16);
	public void RestoreFrom(ValueInstance[] snapshot) => Array.Copy(snapshot, data, 16);
	public void Clear() => Array.Clear(data, 0, 16);
}