namespace Strict.Tests;

public class RunnerTests
{
	[Test]
	public void RunSimpleCalculator()
	{
		using var writer = new StringWriter();
		var rememberConsole = Console.Out;
		Console.SetOut(writer);
		try
		{
			new Runner("Examples/SimpleCalculator.strict").Run();
		}
		finally
		{
			Console.SetOut(rememberConsole);
		}
		Assert.That(writer.ToString(), Is.EqualTo(
			"2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6" + Environment.NewLine));
	}
}