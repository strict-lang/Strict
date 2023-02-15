namespace Strict.VirtualMachine;

public class Registry
{
	public int NextRegister { get; set; }
	public Register PreviousRegister { get; set; }
	private readonly Register[] registers = Enum.GetValues<Register>();
	private readonly List<Register> lockedRegisters = new();


	public Register AllocateRegister(bool isLocked = false)
	{
		if (NextRegister == registers.Length)
			NextRegister = 0;
		PreviousRegister = registers[NextRegister];
		var currentRegister = registers[NextRegister++];
		if (lockedRegisters.Contains(currentRegister))
			currentRegister = AllocateRegister();
		if (isLocked)
			lockedRegisters.Add(currentRegister);
		return currentRegister;
	}

	public void FreeRegisters() => lockedRegisters.Clear();
}