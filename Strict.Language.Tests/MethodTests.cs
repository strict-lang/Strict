using NUnit.Framework;
using Strict.Language.Expressions;
using static Strict.Language.NamedType;

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
			() => new Method(type, 0, null!, new[] { "ab(" }));

	[Test]
	public void ParametersMustNotBeEmpty() =>
		Assert.That(() => new Method(type, 0, null!, new[] { "ab()" }),
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
	public const string LetNumber = "	constant number = 5";
	public const string LetOther = "	constant other = 3";
	public const string LetErrorMessage = "\tconstant errorMessage = \"some error\"";

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
			"	constant result = variable + \"5\""
		});
		Assert.That(method.Name, Is.EqualTo(Run));
		Assert.That(method.Parameters, Has.Count.EqualTo(1));
		var binary = (Binary)((ConstantDeclaration)method.GetBodyAndParseIfNeeded()).Value;
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

	[Test]
	public void CloningWithSameParameterType()
	{
		var method = new Method(type, 0, new MethodExpressionParser(), new[]
		{
			"Run(variable Text)",
			"	\"5\""
		});
		Assert.That(method.Parameters[0].CloneWithImplementationType(type.GetType(Base.Text)), Is.EqualTo(method.Parameters[0]));
	}

	[Test]
	public void SplitTestExpressions()
	{
		var customType = new Type(new TestPackage(),
			new TypeLines(nameof(SplitTestExpressions),
				"has log",
				"AddFive(variable Text) Text",
				"	AddFive(\"5\") is \"55\"",
				"	AddFive(\"6\") is \"65\"",
				"	variable + \"5\"")).ParseMembersAndMethods(new MethodExpressionParser());
		customType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(customType.Methods[0].Tests.Count, Is.EqualTo(2));
	}

	[Test]
	public void ConditionalExpressionIsNotTest()
	{
		var customType = new Type(new TestPackage(),
			new TypeLines(nameof(SplitTestExpressions),
				"has log",
				"ConditionalExpressionIsNotTest Boolean",
				"	5 is 5 ? true else false")).ParseMembersAndMethods(new MethodExpressionParser());
		customType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(customType.Methods[0].Tests.Count, Is.EqualTo(0));
	}

	[Test]
	public void MethodParameterWithDefaultValue()
	{
		var method = new Method(type, 0, new MethodExpressionParser(), new[]
		{
			"Run(input = \"Hello\")",
			"	\"5\""
		});
		Assert.That(method.Parameters[0].DefaultValue, Is.EqualTo(new Text(type, "Hello")));
	}

	[Test]
	public void TraitMethodParameterCannotHaveDefaultValue() =>
		Assert.That(
			() => new Method(type, 0, new MethodExpressionParser(), new[] { "Run(input = \"Hello\")" }),
			Throws.InstanceOf<Method.DefaultValueCouldNotBeParsedIntoExpression>()!);

	[Test]
	public void ImmutableMethodParameterValueCannotBeChanged()
	{
		var method = new Method(type, 0, new MethodExpressionParser(), new[]
		{
			"Run(input = \"Hello\")",
			"	input = \"Hi\""
		});
		Assert.That(() => method.GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());
	}

	[Test]
	public void ImmutableMethodVariablesCannotBeChanged()
	{
		var method = new Method(type, 0, new MethodExpressionParser(), new[]
		{
			"Run",
			"	constant random = \"Hi\"",
			"	random = 5"
		});
		Assert.That(() => method.GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());
	}

	[Test]
	public void ValueTypeNotMatchingWithAssignmentType() =>
		Assert.That(
			() => new Method(type, 0, new MethodExpressionParser(),
				new[] { "Run(mutable input = 0)", "	input = \"5\"" }).GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MutableAssignment.ValueTypeNotMatchingWithAssignmentType>());

	[Test]
	public void MissingParameterDefaultValue() =>
		Assert.That(
			() => new Method(type, 0, new MethodExpressionParser(),
				new[] { "Run(input =)", "	5" }),
			Throws.InstanceOf<Method.MissingParameterDefaultValue>());

	[Test]
	public void ParameterWithTypeNameAndInitializerIsForbidden() =>
		Assert.That(
			() => new Method(type, 0, new MethodExpressionParser(),
				new[] { "Run(input Number = 5)", "	5" }),
			Throws.InstanceOf<AssignmentWithInitializerTypeShouldNotHaveNameWithType>());
}