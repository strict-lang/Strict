namespace TestPackage;

public class ExecuteOperation
{
	private Register firstRegister;
	private Register secondRegister;
	public Register Add()
	{
		firstRegister + secondRegister;
	}
	public Register Subtract()
	{
		firstRegister - secondRegister;
	}

	[Test]
	public void AddTest()
	{
		Assert.That(() => new ExecuteOperation(new Register(1, 0, 0), new Register(2, 0, 0)).Add() == new Register(3, 0, 0)));
	[Test]
	public void SubtractTest()
	{
		Assert.That(() => new ExecuteOperation(new Register(3, 0, 0), new Register(1, 0, 0)).Subtract() == new Register(2, 0, 0)));
	}
}