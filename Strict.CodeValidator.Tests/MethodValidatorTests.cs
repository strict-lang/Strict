using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Tests;
using static Strict.CodeValidator.MethodValidator;
using Type = Strict.Language.Type;

namespace Strict.CodeValidator.Tests;

public sealed class MethodValidatorTests
{
	[SetUp]
	public void CreateTypeAndParser()
	{
		type = new Type(new TestPackage(), new TypeLines(nameof(MethodValidatorTests), ""));
		parser = new MethodExpressionParser();
	}

	private Type type = null!;
	private ExpressionParser parser = null!;

	[Test]
	public void UnchangedMutableVariablesShouldError() =>
		Assert.That(
			() => new MethodValidator(new[]
			{
				new Method(type, 1, parser, new[]
				{
					"Run",
					"\tmutable input = 0",
					"\tinput + 5"
				})
			}).Validate(),
			Throws.InstanceOf<VariableDeclaredAsMutableButValueNeverChanged>().With.
				Message.Contains("input"));

	[Test]
	public void ExceptionShouldOccurOnlyForUnchangedMutableVariable() =>
		Assert.That(
			() => new MethodValidator(new[]
			{
				new Method(type, 1, parser, new[]
				{
					"Run",
					"\tmutable inputOne = 0",
					"\tinputOne = 5",
					"\tmutable inputTwo = 0",
					"\tinputTwo = 6",
					"\tmutable inputThree = 0",
					"\tinputOne + inputTwo + inputThree"
				})
			}).Validate(),
			Throws.InstanceOf<VariableDeclaredAsMutableButValueNeverChanged>().With.
				Message.Contains("inputThree"));

	[Test]
	public void ConstantVariablesShouldBeAllowedToPass() =>
		Assert.DoesNotThrow(() => new MethodValidator(new[]
		{
			new Method(type, 1, parser, new[]
			{
				"Run",
				"\tconstant input = 10",
				"\tinput + 5"
			})
		}).Validate());

	[Test]
	public void MutatedVariablesShouldBeAllowedToPass() =>
		Assert.DoesNotThrow(() => new MethodValidator(new[]
		{
			new Method(type, 1, parser, new[]
			{
				"Run",
				"\tmutable input = 10",
				"\tinput = 15",
				"\tinput + 15"
			})
		}).Validate());

	[TestCase("methodInput", "Run(methodInput Number)", "\t\"Run method executed\"")]
	[TestCase("second", "Run(first Number, second Text)", "\tconstant result = first + 5",
		"\t\"Run method executed\"")]
	public void ValidateUnusedMethodParameter(string expectedOutput, params string[] methodLines) =>
		Assert.That(
			() => new MethodValidator(new[] { new Method(type, 1, parser, methodLines) }).Validate(),
			Throws.InstanceOf<UnusedMethodParameterMustBeRemoved>().With.Message.
				Contains(expectedOutput));

	[Test]
	public void ErrorOnlyIfParametersAreUnused() =>
		Assert.DoesNotThrow(() => new MethodValidator(new[]
		{
			new Method(type, 1, parser,
				new[]
				{
					"Run(methodInput Number)",
					"\t\"Run method executed with input\" + methodInput"
				})
		}).Validate());

	[TestCase("Run(mutable parameter Number)", "\tconstant result = 5 + parameter", "\tresult")]
	[TestCase("Run(mutable mutatedParameter Number, mutable parameter Number)", "\tmutatedParameter = 5 + parameter", "\tmutatedParameter")]
	public void UnchangedMutableParametersShouldError(params string[] code) =>
		Assert.That(() => new MethodValidator(new[] { new Method(type, 1, parser, code) }).Validate(),
			Throws.InstanceOf<ParameterDeclaredAsMutableButValueNeverChanged>().With.Message.
				Contains("parameter"));

	[Test]
	public void MutatedParametersShouldBeAllowed() =>
		Assert.DoesNotThrow(
			() => new MethodValidator(new[]
			{
				new Method(type, 1, parser,
					new[]
					{
						"Run(mutable parameter Number)",
						"\tparameter = 5 + parameter",
						"\t5"
					})
			}).Validate());
}