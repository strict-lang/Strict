using NUnit.Framework;

namespace Strict.VirtualMachine.Tests;

public class VirtualMachineTestsKata : VirtualMachineTests
{
	[Test]
	public void BestTimeToBuyStocksKata()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("Stock",
			"Stock((7, 1, 5, 3, 6, 4)).MaxProfit", "has prices Numbers", "MaxProfit Number",
			"\tmutable min = 999", //Need to implement int.MaxValue
			"\tmutable max = 0", "\tfor prices", "\t\tif value < min", "\t\t\tmin = value",
			"\t\telse if value - min > max", "\t\t\tmax = value - min", "\tmax")).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(5));
	}
}