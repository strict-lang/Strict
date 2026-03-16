using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Bytecode.Serialization;

public sealed record BytecodeMember(string Name, string FullTypeName,
	Instruction? InitialValueExpression)
{
	public BytecodeMember(BinaryReader reader, NameTable table, StrictBinary binary)
		: this(table.Names[reader.Read7BitEncodedInt()], table.Names[reader.Read7BitEncodedInt()],
			reader.ReadBoolean()
				? binary.ReadInstruction(reader, table)
				: null)	{ }

	public string JustTypeName => FullTypeName.Split(Context.ParentSeparator)[^1];

	public override string ToString() =>
		Name + " " + JustTypeName + (InitialValueExpression != null
			? " = " + InitialValueExpression
			: "");

	public void Write(BinaryWriter writer, NameTable table)
	{
		writer.Write7BitEncodedInt(table[Name]);
		writer.Write7BitEncodedInt(table[FullTypeName]);
		writer.Write(InitialValueExpression != null);
		if (InitialValueExpression != null)
			InitialValueExpression.Write(writer, table);
	}
}