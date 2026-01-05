using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.TestRunner.Tests;

public sealed class TestExecutorTests
{
	[SetUp]
	public void Setup() => executor = new TestExecutor(TestPackage.Instance);

	private TestExecutor executor = null!;

	[Test]
	public void RunExampleTest()
	{
		var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunExampleTest),
				"has number",
				"Run Number",
				"	5 is 5",
				"	10")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.RunMethod(type.Methods.First(m => m.Name == "Run")), Is.True);
	}

	[Test]
	public void RunExampleTestFails()
	{
		var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunExampleTestFails),
				"has number",
				"Run Number",
				"	5 is 6",
				"	10")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.RunMethod(type.Methods.First(m => m.Name == "Run")), Is.False);
	}

	[Test]
	public void RunTestsForType()
	{
		var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunTestsForType),
				"has number",
				"Run Number",
				"	5 is 5",
				"	10",
				"Other Number",
				"	2 is 2",
				"	5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.RunTests(type), Is.True);
	}

	[Test]
	public void RunTestsForTypeFails()
	{
		var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunTestsForTypeFails),
				"has number",
				"Run Number",
				"	5 is 5",
				"	10",
				"Other Number",
				"	2 is 3",
				"	5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.RunTests(type), Is.False);
	}
}