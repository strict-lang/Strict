using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public sealed class MethodTests
{
	[SetUp]
	public void CreateType() => type = new Type(new TestPackage(), new MockRunTypeLines());

	private Type type = null!;

	[Test]
	public void MustMustHaveAValidName() =>
		Assert.That(() => new Method(type, 0, null!, new[] { "5(text)" }),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void ReturnTypeMustBeBeLast() =>
		Assert.That(() => new Method(type, 0, null!, new[] { "Texts GetFiles" }),
			Throws.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void InvalidMethodParameters() =>
		Assert.Throws<Method.InvalidMethodParameters>(
			() => new Method(type, 0, null!, new[] { "a(" }));

	[Test]
	public void ParametersMustNotBeEmpty() =>
		Assert.That(() => new Method(type, 0, null!, new[] { "a()" }),
			Throws.InstanceOf<Method.EmptyParametersMustBeRemoved>());

	[TestCase("from(Text)")]
	[TestCase("from(Number)")]
	[TestCase("from(Start Number, End Number)")]
	[TestCase("from(start Number, End Number)")]
	public void UpperCaseParameterWithNoTypeSpecificationIsNotAllowed(string method) =>
		Assert.That(() => new Method(type, 0, null!, new[] { method }),
			Throws.InstanceOf<Method.ParametersMustStartWithLowerCase>());

	[Test]
	public void ParseDefinition()
	{
		var method = new Method(type, 0, null!, new[] { Run });
		Assert.That(method.Name, Is.EqualTo(Run));
		Assert.That(method.Parameters, Is.Empty);
		Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.None)));
		Assert.That(method.ToString(), Is.EqualTo(Run));
	}

	[Test]
	public void ParseFrom()
	{
		var method = new Method(type, 0, null!, new[] { "from(number)" });
		Assert.That(method.Name, Is.EqualTo("from"));
		Assert.That(method.Parameters, Has.Count.EqualTo(1), method.Parameters.ToWordList());
		Assert.That(method.Parameters[0].Type, Is.EqualTo(type.GetType("Number")));
		Assert.That(method.ReturnType, Is.EqualTo(type));
	}

	public const string Run = nameof(Run);

	[Test]
	public void ParseWithReturnType()
	{
		var method = new Method(type, 0, null!, NestedMethodLines);
		Assert.That(method.Name, Is.EqualTo("IsBlaFive"));
		Assert.That(method.Parameters, Is.Empty);
		Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.Boolean)));
		Assert.That(method.ToString(), Is.EqualTo(NestedMethodLines[0]));
	}

	public static readonly string[] NestedMethodLines =
	{
		"IsBlaFive Boolean",
		LetNumber,
		"	if bla is 5",
		"		return true",
		"	false"
	};
	public const string LetNumber = "	let number = 5";
	public const string LetOther = "	let other = 3";
	public const string LetErrorMessage = "\tlet errorMessage = \"some error\"";

	[Test]
	public void TraitMethodBodiesShouldNotBeCalled()
	{
		var appTrait =
			new Type(type.Package, new TypeLines("DummyApp", "Run")).ParseMembersAndMethods(null!);
		Assert.That(() => appTrait.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Method.CannotCallBodyOnTraitMethod>());
	}

	[Test]
	public void AccessValidMethodParametersInMethodBody()
	{
		var method = new Method(type, 0, new MethodExpressionParser(), new[]
		{
			"Run(variable Text)",
			"	let result = variable + \"5\""
		});
		Assert.That(method.Name, Is.EqualTo(Run));
		Assert.That(method.Parameters, Has.Count.EqualTo(1));
		var binary = (Binary)((Assignment)method.GetBodyAndParseIfNeeded()).Value;
		Assert.That(binary.Instance, Is.InstanceOf<ParameterCall>());
	}

	[TestCase("Run(variable Generic)")]
	[TestCase("Run(generic)")]
	[TestCase("Run(number, input Generic, generic)")]
	[TestCase("Run(number) Generic")]
	public void GenericMethods(string methodHeader) =>
		Assert.That(new Method(type, 0, new MethodExpressionParser(), new[]
		{
			methodHeader
		}).IsGeneric, Is.True);

	[TestCase("Run(text) Number")]
	[TestCase("Run(variable Number, input Text) Boolean")]
	public void NonGenericMethods(string methodHeader) =>
		Assert.That(new Method(type, 0, new MethodExpressionParser(), new[]
		{
			methodHeader
		}).IsGeneric, Is.False);
}