namespace Strict.Runtime.Tests;

public sealed class ValueInstanceMigrationTests : BaseVirtualMachineTests
{
	[Test]
	public void StatementsShouldUseValueInstanceNotInstance()
	{
		var valueInstance = new Strict.Expressions.ValueInstance(NumberType, 42.0);
		var statement = new LoadConstantStatement(Register.R0, valueInstance);
		Assert.That(statement.ValueInstance, Is.EqualTo(valueInstance));
		Assert.That(statement.ValueInstance.Number, Is.EqualTo(42.0));
	}
}
