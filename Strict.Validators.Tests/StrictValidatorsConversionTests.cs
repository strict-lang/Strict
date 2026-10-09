namespace Strict.Validators.Tests;

/// <summary>
/// Guards for converting Strict.Validators C# types to .strict files under Validators/.
/// </summary>
public sealed class StrictValidatorsConversionTests
{
	[Test]
	public void ValidatorsPackageHasExpectedTypes()
	{
		var path = GetValidatorsPath();
		foreach (var typeName in new[]
			{
				"ValidationIssue", "UsageRules", "TypeValidator", "ConstantCollapser", "ValidateCheck"
			})
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True, typeName);
	}

	[Test]
	public async Task LoadTypeValidatorFromValidatorsPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Validators");
		var typeValidator = package.GetType("TypeValidator");
		Assert.That(typeValidator.Methods.Any(method => method.Name == "Validate"), Is.True);
		Assert.That(typeValidator.Methods.Any(method => method.Name == "MemberIssues"), Is.True);
		Assert.That(typeValidator.Methods.Any(method => method.Name == "IsExempt"), Is.True);
	}

	[Test]
	public async Task LoadConstantCollapserFromValidatorsPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Validators");
		var collapser = package.GetType("ConstantCollapser");
		Assert.That(collapser.Methods.Any(method => method.Name == "CollapseBinary"), Is.True);
		Assert.That(collapser.Methods.Any(method => method.Name == "CollapseTo"), Is.True);
		Assert.That(collapser.Methods.Any(method => method.Name == "ShouldUseConstant"), Is.True);
	}

	[Test]
	public async Task LoadUsageRulesFromValidatorsPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Validators");
		var usageRules = package.GetType("UsageRules");
		Assert.That(usageRules.Members.Count, Is.EqualTo(4));
		Assert.That(usageRules.Methods.Any(method => method.Name == "IsChanged"), Is.True);
	}

	private static string GetValidatorsPath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Validators");
}