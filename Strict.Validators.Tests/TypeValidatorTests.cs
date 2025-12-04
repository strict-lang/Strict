global using Type = Strict.Language.Type;

namespace Strict.Validators.Tests;

public sealed class TypeValidatorTests
{
//TODO: fix
/*
	[SetUp]
	public void CreateTypeAndParser()
	{
		type = new Type(TestPackage.Instance,
			new TypeLines(nameof(TypeValidatorTests), "has logger", "Run", "\tlogger.Log(5)"));
		parser = new MethodExpressionParser();
		type.ParseMembersAndMethods(parser);
		validator = new TypeValidator();
	}

	private Type type = null!;
	private ExpressionParser parser = null!;
	private TypeValidator validator = null!;

	[TearDown]
	public void TearDown() => type.Dispose();

	[TestCase("unused", "Run", "\tconstant unused = \"something never used\"",
		"\t\"Run method executed\"")]
	[TestCase("secondIsUnused", "Run(input Text)", "\tconstant first = input + 5",
		"\tconstant secondIsUnused = input + 5", "\tfirst + \"Run method executed\"")]
	public void ValidateUnusedMethodVariables(string expectedOutput, params string[] methodLines) =>
		Assert.That(
			() => validator.Visit(new Method(type, 1, parser, methodLines), true),
			Throws.InstanceOf<TypeValidator.UnusedMethodVariableMustBeRemoved>().With.Message.
				Contains(expectedOutput));

	[Test]
	public void ErrorOnlyIfVariablesAreUnused() =>
		Assert.DoesNotThrow(() => validator.Visit(
			new Method(type, 1, parser, [
				"Run(methodInput Number)",
				"\tconstant result = 5 + 15 + methodInput",
				"\t\"Run method executed with input\" + result"
			]), true));

	[Test]
	public void UnchangedMutableVariablesShouldError() =>
		Assert.That(() => validator.Visit(new Method(type, 1, parser, [
				"Run",
				"\tmutable input = 0",
				"\tinput + 5"
			]), true),
			Throws.InstanceOf<TypeValidator.VariableDeclaredAsMutableButValueNeverChanged>().With.
				Message.Contains("input"));

	[Test]
	public void ExceptionShouldOccurOnlyForUnchangedMutableVariable() =>
		Assert.That(() => validator.Visit(new Method(type, 1, parser, [
				"Run",
				"\tmutable inputOne = 0",
				"\tinputOne = 5",
				"\tmutable inputTwo = 0",
				"\tinputTwo = 6",
				"\tmutable inputThree = 0",
				"\tinputOne + inputTwo + inputThree"
			]), true),
			Throws.InstanceOf<TypeValidator.VariableDeclaredAsMutableButValueNeverChanged>().With.
				Message.Contains("inputThree"));

	[Test]
	public void ConstantVariablesShouldBeAllowedToPass() =>
		Assert.DoesNotThrow(() => validator.Visit(
			new Method(type, 1, parser, [
				"Run",
				"\tconstant input = 10",
				"\tinput + 5"
			]), true));

	[Test]
	public void MutatedVariablesShouldBeAllowedToPass() =>
		Assert.DoesNotThrow(() => validator.Visit(
			new Method(type, 1, parser, [
				"Run",
				"\tmutable input = 10",
				"\tinput = 15",
				"\tinput + 15"
			]), true));

	[TestCase("methodInput", "Run(methodInput Number)", "\t\"Run method executed\"")]
	[TestCase("second", "Run(first Number, second Text)", "\tconstant result = first + 5",
		"\t\"Run method executed\" + result")]
	public void ValidateUnusedMethodParameter(string expectedOutput, params string[] methodLines) =>
		Assert.That(
			() => validator.Visit(new Method(type, 1, parser, methodLines), true),
			Throws.InstanceOf<TypeValidator.UnusedMethodParameterMustBeRemoved>().With.Message.
				Contains(expectedOutput));

	[Test]
	public void ErrorOnlyIfParametersAreUnused() =>
		Assert.DoesNotThrow(() => validator.Visit(
			new Method(type, 1, parser, [
				"Run(methodInput Number)",
				"\t\"Run method executed with input\" + methodInput"
			]), true));

	[TestCase("Run(mutable parameter Number)", "\tconstant result = 5 + parameter", "\tresult")]
	[TestCase("Run(mutable otherMutatedParameter Number, mutable parameter Number)",
		"\totherMutatedParameter = 5 + parameter", "\totherMutatedParameter")]
	public void UnchangedMutableParametersShouldError(params string[] code) =>
		Assert.That(() => validator.Visit(new Method(type, 1, parser, code), true),
			Throws.InstanceOf<TypeValidator.ParameterDeclaredAsMutableButValueNeverChanged>().With.
				Message.Contains("parameter"));

	[Test]
	public void MutatedParametersShouldBeAllowed() =>
		Assert.DoesNotThrow(() => validator.Visit(
			new Method(type, 1, parser, [
				"Run(mutable parameter Number)", "\tparameter = 5 + parameter", "\t5"
			]), true));

	[Test]
	public void ListArgumentCanBeAutoParsedWithoutDoubleBrackets()
	{
		using var typeWithListParameterMethod = new Type(TestPackage.Instance,
			new TypeLines(nameof(ListArgumentCanBeAutoParsedWithoutDoubleBrackets), "has logger",
				"CheckInputLengthAndGetResult(numbers) Number", "\tif numbers.Length is 2",
				"\t\treturn 2", "\t0")).ParseMembersAndMethods(parser);
		Assert.That(() => validator.Visit(new Method(typeWithListParameterMethod, 1, parser, [
				"InvokeTestMethod(numbers) Number",
				"\tif numbers.Length is 2",
				"\t\treturn 2",
				"\tCheckInputLengthAndGetResult((1, 2))"
			]), true),
			Throws.InstanceOf<TypeValidator.ListArgumentCanBeAutoParsedWithoutDoubleBrackets>().With.
				Message.Contains("CheckInputLengthAndGetResult((1, 2))"));
	}

	[Test]
	public void ValidateUnusedMember() =>
		Assert.That(() => validator.Visit(CreateType(nameof(ValidateUnusedMember), [
				"has unused Number", "Run(methodInput Number)",
				"\tconstant result = 5 + methodInput", "\tresult"
			])),
			Throws.InstanceOf<TypeValidator.UnusedMemberMustBeRemoved>().With.Message.
				Contains("unused"));

	private Type CreateType(string typeName, string[] code) =>
		new Type(type.Package, new TypeLines(typeName, code)).ParseMembersAndMethods(parser);

	[Test]
	public void ProperlyUsedMemberShouldBeAllowed() =>
		Assert.That(() => validator.Visit(CreateType(nameof(ProperlyUsedMemberShouldBeAllowed), [
			"has usedMember Number",
			"Run(methodInput Number)",
			"\tconstant result = usedMember + methodInput",
			"\tresult"
		])), Throws.Nothing);

	[Test]
	public void ValidateTypeHasTooManyDependenciesFromMethod() =>
		Assert.That(() => validator.Visit(CreateType(
				nameof(ValidateTypeHasTooManyDependenciesFromMethod), [
					"has number",
					"from(number, text, boolean, input Text, another Number)",
					"\tvalue",
					"Run(methodInput Number)",
					"\tif boolean",
					"\t\treturn text + input + number + methodInput + character",
					"\t0"
				])),
			Throws.InstanceOf<Method.MethodParameterCountMustNotExceedLimit>().With.Message.Contains(
				"Type TestPackage.ValidateTypeHasTooManyDependenciesFromMethod from constructor method " +
				"has parameters count 5 but limit is 4"));

	[Test]
	public void VariableHidesMemberUseDifferentName() =>
		Assert.That(() =>
			{
				using var typeWithInputMember = CreateType(nameof(VariableHidesMemberUseDifferentName), [
					"has input Number",
					"FirstMethod(methodInput Number) Number",
					"\tconstant something = 5",
					"\tmethodInput + something",
					"SecondMethod(methodInput Number) Number",
					"\tconstant second = 5",
					"\tmethodInput + second",
					"Run(methodInput Number)",
					"\tconstant input = 5",
					"\tmethodInput + input"
				]);
				typeWithInputMember.Methods[^1].GetBodyAndParseIfNeeded();
				validator.Visit(typeWithInputMember);
			},
			Throws.InstanceOf<TypeValidator.VariableHidesMemberUseDifferentName>().With.Message.
				Contains("Method name Run, Variable name input"));

	[Test]
	public void ParameterHidesMemberUseDifferentName() =>
		Assert.That(() => validator.Visit(CreateType(nameof(ParameterHidesMemberUseDifferentName), [
				"has input Number",
				"FirstMethod(input Number) Number",
				"\tconstant something = 5",
				"\tinput + something",
				"SecondMethod(methodInput Number) Number",
				"\tconstant second = 5",
				"\tmethodInput + second"
			])),
			Throws.InstanceOf<TypeValidator.ParameterHidesMemberUseDifferentName>().With.Message.
				Contains("Method name FirstMethod, Parameter name input"));

	[Test]
	public void CheckPackage() => Assert.That(() => validator.Visit(type.Package), Throws.Nothing);
*/
}