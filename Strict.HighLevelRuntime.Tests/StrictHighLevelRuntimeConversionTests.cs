using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.HighLevelRuntime.Tests;

/// <summary>
/// Guards for converting Strict.HighLevelRuntime C# types to .strict files under HighLevelRuntime/.
/// </summary>
public sealed class StrictHighLevelRuntimeConversionTests
{
	[Test]
	public void HighLevelRuntimePackageHasExpectedTypes()
	{
		var path = GetHighLevelRuntimePath();
		foreach (var typeName in new[]
		{
			"RuntimeStatistics", "TestBehavior", "RuntimeValue", "ExecutionContext", "BodyResult",
			"IfEvaluator", "ForEvaluator", "ToEvaluator", "SelectorIfEvaluator", "MethodCallEvaluator",
			"ExpressionEvaluator", "BodyEvaluator", "Evaluators", "Interpreter"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadInterpreterFromHighLevelRuntimePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage(
				"Strict/HighLevelRuntime");
		var interpreter = package.GetType("Interpreter");
		Assert.That(interpreter.Methods.Any(method => method.Name == "EvaluateLine"), Is.True);
		Assert.That(interpreter.Methods.Any(method => method.Name == "EvaluateBody"), Is.True);
		Assert.That(interpreter.Methods.Any(method => method.Name == "Empty"), Is.True);
	}

	[Test]
	public async Task LoadExpressionEvaluatorFromHighLevelRuntimePackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage(
				"Strict/HighLevelRuntime");
		var evaluator = package.GetType("ExpressionEvaluator");
		Assert.That(evaluator.Methods.Any(method => method.Name == "Evaluate"), Is.True);
		Assert.That(evaluator.Methods.Any(method => method.Name == "IsNumberLiteral"), Is.True);
	}

	[Test]
	public async Task LoadRuntimeValueAndMethodCallEvaluator()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage(
				"Strict/HighLevelRuntime");
		var value = package.GetType("RuntimeValue");
		Assert.That(value.Methods.Any(method => method.Name == "FromNumber"), Is.True);
		Assert.That(value.Methods.Any(method => method.Name == "EqualsValue"), Is.True);
		var callers = package.GetType("MethodCallEvaluator");
		Assert.That(callers.Methods.Any(method => method.Name == "EvaluateBinary"), Is.True);
		Assert.That(callers.Methods.Any(method => method.Name == "EvaluateNot"), Is.True);
	}

	[Test]
	public async Task LoadExecutionContextAndStatistics()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage(
				"Strict/HighLevelRuntime");
		var context = package.GetType("ExecutionContext");
		Assert.That(context.Methods.Any(method => method.Name == "SetValue"), Is.True);
		Assert.That(context.Methods.Any(method => method.Name == "GetValue"), Is.True);
		var stats = package.GetType("RuntimeStatistics");
		Assert.That(stats.Methods.Any(method => method.Name == "IncrementExpressions"), Is.True);
		Assert.That(stats.Methods.Any(method => method.Name == "Empty"), Is.True);
	}

	[Test]
	public void HighLevelRuntimePackageHasDemoAndTestEntryPoints()
	{
		var path = GetHighLevelRuntimePath();
		foreach (var typeName in new[]
		{
			"RuntimeDemo", "RuntimeValueTests", "EvaluatorTests", "IfToTests", "ContextTests",
			"InterpreterTests", "BodyTests"
		})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadMethodCallEvaluatorArithmeticSurface()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage(
				"Strict/HighLevelRuntime");
		var callers = package.GetType("MethodCallEvaluator");
		foreach (var name in new[] { "Add", "Subtract", "Multiply", "Divide", "EvaluateCompare" })
			Assert.That(callers.Methods.Any(method => method.Name == name), Is.True, name);
	}

	private static string GetHighLevelRuntimePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"HighLevelRuntime");
}
