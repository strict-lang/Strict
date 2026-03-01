namespace Strict.Runtime;

public sealed class Registry
{
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