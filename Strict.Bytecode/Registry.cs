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

	public int NextRegister { get; private set; }
	public Register PreviousRegister { get; set; }

	public Register AllocateRegister()
	{
		if (NextRegister >= Registers.Count)
			throw new OutOfRegisters(Registers.Count);
		PreviousRegister = (Register)NextRegister;
		return (Register)NextRegister++;
	}

	/// <summary>
	/// Thrown when a single method body needs more virtual registers than available.
	/// Prefer splitting the method over silent wrap-around (which corrupts live values).
	/// </summary>
	public sealed class OutOfRegisters(int limit) : Exception(
		"Bytecode method exhausted all " + limit +
		" virtual registers; simplify the method or increase Registers.Count");
}
