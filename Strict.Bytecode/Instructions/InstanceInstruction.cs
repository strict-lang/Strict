using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Instructions;

public abstract class InstanceInstruction(InstructionType instructionType,
	ValueInstance valueInstance) : Instruction(instructionType)
{
	public ValueInstance ValueInstance { get; } = valueInstance;
	public override string ToString() => $"{InstructionType} {ValueInstance.ToExpressionCodeString()}";

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		WriteValueInstance(writer, ValueInstance, table);
	}

	internal static void WriteValueInstance(BinaryWriter writer, ValueInstance val, NameTable table)
	{
		if (val.IsText)
		{
			writer.Write((byte)ValueKind.Text);
			writer.Write7BitEncodedInt(table[val.Text]);
			return;
		}
		if (val.IsList)
		{
			writer.Write((byte)ValueKind.List);
			writer.Write7BitEncodedInt(table[val.List.ReturnType.Name]);
			var items = val.List.Items;
			writer.Write7BitEncodedInt(items.Count);
			foreach (var item in items)
				WriteValueInstance(writer, item, table);
			return;
		}
		if (val.IsDictionary)
		{
			writer.Write((byte)ValueKind.Dictionary);
			writer.Write7BitEncodedInt(table[val.GetType().Name]);
			var items = val.GetDictionaryItems();
			writer.Write7BitEncodedInt(items.Count);
			foreach (var kvp in items)
			{
				WriteValueInstance(writer, kvp.Key, table);
				WriteValueInstance(writer, kvp.Value, table);
			}
			return;
		}
		var type = val.GetType();
		if (type.IsBoolean)
		{
			writer.Write((byte)ValueKind.Boolean);
			writer.Write(val.Boolean);
			return;
		}
		if (type.IsNone)
		{
			writer.Write((byte)ValueKind.None);
			return;
		}
		if (type.IsNumber)
		{
			if (IsSmallNumber(val.Number))
			{
				writer.Write((byte)ValueKind.SmallNumber);
				writer.Write((byte)(int)val.Number);
			}
			else if (IsIntegerNumber(val.Number))
			{
				writer.Write((byte)ValueKind.IntegerNumber);
				writer.Write((int)val.Number);
			}
			else
			{
				writer.Write((byte)ValueKind.Number);
				writer.Write(val.Number);
			}
		}
		else
			throw new NotSupportedException( //ncrunch: no coverage
				"WriteValueInstance not supported value: " + val);
	}

	public static bool IsSmallNumber(double value) =>
		value is >= 0 and <= 255 && value == Math.Floor(value);

	public static bool IsIntegerNumber(double value) =>
		value is >= int.MinValue and <= int.MaxValue && value == Math.Floor(value);
}