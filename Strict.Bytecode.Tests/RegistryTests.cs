using Strict.Bytecode;

namespace Strict.Bytecode.Tests;

public sealed class RegistryTests
{
	[Test]
	public void AllocatesSequentialRegistersWithoutWrap()
	{
		var registry = new Registry();
		Assert.That(registry.AllocateRegister(), Is.EqualTo(Register.R0));
		Assert.That(registry.AllocateRegister(), Is.EqualTo(Register.R1));
		Assert.That(registry.PreviousRegister, Is.EqualTo(Register.R1));
	}

	[Test]
	public void AllocatesAllRegistersThenThrowsInsteadOfWrapping()
	{
		var registry = new Registry();
		for (var index = 0; index < Registers.Count; index++)
			Assert.That((int)registry.AllocateRegister(), Is.EqualTo(index));
		Assert.Throws<Registry.OutOfRegisters>(() => registry.AllocateRegister());
	}

	[Test]
	public void RegisterCountIsAtLeastSixtyFour() =>
		Assert.That(Registers.Count, Is.GreaterThanOrEqualTo(64));
}
