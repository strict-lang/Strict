using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.TestRunner.Tests;

/// <summary>
/// Guards for converting Strict.TestRunner C# types to .strict files under TestRunner/.
/// </summary>
public sealed class StrictTestRunnerConversionTests
{
	[Test]
	public void TestRunnerPackageHasExpectedTypes()
	{
		var path = GetTestRunnerPath();
		foreach (var typeName in new[]
		{
			"TestStatistics", "TestResult", "Assertion", "MethodUnderTest", "TypeUnderTest",
			"TestInterpreter"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadTestInterpreterFromTestRunnerPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/TestRunner");
		var interpreter = package.GetType("TestInterpreter");
		Assert.That(interpreter.Methods.Any(method => method.Name == "RunMethod"), Is.True);
		Assert.That(interpreter.Methods.Any(method => method.Name == "RunAllTestsInType"), Is.True);
		Assert.That(interpreter.Methods.Any(method => method.Name == "RunAssertion"), Is.True);
		Assert.That(interpreter.Methods.Any(method => method.Name == "AllPassed"), Is.True);
	}

	[Test]
	public async Task LoadAssertionFromTestRunnerPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/TestRunner");
		var assertion = package.GetType("Assertion");
		Assert.That(assertion.Methods.Any(method => method.Name == "Evaluate"), Is.True);
		Assert.That(assertion.Methods.Any(method => method.Name == "IsAssertion"), Is.True);
		Assert.That(assertion.Methods.Any(method => method.Name == "IsNotAssertion"), Is.True);
	}

	[Test]
	public async Task LoadTestStatisticsFromTestRunnerPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/TestRunner");
		var stats = package.GetType("TestStatistics");
		Assert.That(stats.Methods.Any(method => method.Name == "IncrementMethods"), Is.True);
		Assert.That(stats.Methods.Any(method => method.Name == "Empty"), Is.True);
	}

	private static string GetTestRunnerPath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"TestRunner");
}
