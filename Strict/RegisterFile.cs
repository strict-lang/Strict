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
		get
		{
			var value = data[(int)r];
#if DEBUG
			if (Language.PerformanceLog.IsEnabled)
				Language.PerformanceLog.Write("RegisterFile.get", "register=" + r + ", value=" + Describe(value));
#endif
			return value;
		}
		set
		{
#if DEBUG
			if (Language.PerformanceLog.IsEnabled)
				Language.PerformanceLog.Write("RegisterFile.set", "register=" + r + ", value=" + Describe(value));
#endif
			data[(int)r] = value;
		}
	}

	/// <summary>
	/// Returns false (and a default value) only when the slot has never been written.
	/// </summary>
	internal bool TryGet(Register r, out ValueInstance value)
	{
		value = data[(int)r];
#if DEBUG
		if (Language.PerformanceLog.IsEnabled)
			Language.PerformanceLog.Write("RegisterFile.TryGet", "register=" + r + ", value=" + Describe(value));
#endif
		return value.HasValue;
	}

	public void SaveTo(ValueInstance[] snapshot)
	{
#if DEBUG
		if (Language.PerformanceLog.IsEnabled)
			Language.PerformanceLog.Write("RegisterFile.SaveTo", "snapshotLength=" + snapshot.Length);
#endif
		Array.Copy(data, snapshot, 16);
	}

	public void RestoreFrom(ValueInstance[] snapshot)
	{
#if DEBUG
		if (Language.PerformanceLog.IsEnabled)
			Language.PerformanceLog.Write("RegisterFile.RestoreFrom", "snapshotLength=" + snapshot.Length);
#endif
		Array.Copy(snapshot, data, 16);
	}

	public void Clear()
	{
#if DEBUG
		if (Language.PerformanceLog.IsEnabled)
			Language.PerformanceLog.Write("RegisterFile.Clear", "registerCount=16");
#endif
		Array.Clear(data, 0, 16);
	}
#if DEBUG
	private static string Describe(ValueInstance value)
	{
		if (!value.HasValue)
			return "unset";
		if (value.IsText)
			return "Text(length=" + value.Text.Length + ")";
		if (value.IsList)
			return "List(type=" + value.List.ReturnType.Name + ", count=" + value.List.Items.Count + ")";
		if (value.IsDictionary)
			return "Dictionary(count=" + value.GetDictionaryItems().Count + ")";
		var typeInstance = value.TryGetValueTypeInstance();
		return typeInstance != null
			? "TypeInstance(type=" + typeInstance.ReturnType.Name + ", members=" + typeInstance.Values.Length + ")"
			: value.GetType().Name + "(" + value.Number + ")";
	}
#endif
}