namespace Strict.Bytecode;

public sealed class Registry()
{
	public Registry(BinaryReader reader) : this()
	{
		var nextRegisterCount = reader.ReadByte();
		var prev = (Register)reader.ReadByte();
		for (var index = 0; index < nextRegisterCount; index++)
			AllocateRegister();
		PreviousRegister = prev;
	}

	private readonly Register[] registers = Enum.GetValues<Register>();
	public int NextRegister { get; private set; }
	public Register PreviousRegister { get; set; }

	public Register AllocateRegister()
	{
		if (NextRegister == registers.Length)
			NextRegister = 0;
		PreviousRegister = registers[NextRegister];
		return registers[NextRegister++];
	}
}