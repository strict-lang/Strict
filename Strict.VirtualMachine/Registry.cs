namespace Strict.VirtualMachine;

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
		var currentRegister = registers[NextRegister++];
		return currentRegister;
	}
}