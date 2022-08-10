using NUnit.Framework;

namespace Strict.Language.Tests;

public class MethodTests
{
	[SetUp]
	public void CreateType() => type = new Type(new TestPackage(), new MockRunTypeLines());

	private Type type = null!;

	[Test]
	public void MustMustHaveAValidName() =>
		Assert.Throws<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>(() =>
			new Method(type, 0, null!, new[] { "5(text)" }));

	[Test]
	public void ReturnShouldBeReturns() =>
		Assert.Throws<Method.ExpectedReturns>(() => new Method(type, 0, null!, new[] { "GetFiles return Texts" }));

	[Test]
	public void InvalidMethodParameters() =>
		Assert.Throws<Method.InvalidMethodParameters>(() =>
			new Method(type, 0, null!, new[] { "a(" }));

	[Test]
	public void ParametersMustNotBeEmpty() =>
		Assert.Throws<Method.EmptyParametersMustBeRemoved>(() =>
			new Method(type, 0, null!, new[] { "a()" }));

	[Test]
	public void ParseDefinition()
	{
		var method = new Method(type, 0, null!, new[] { Run });
		Assert.That(method.Name, Is.EqualTo(Run));
		Assert.That(method.Parameters, Is.Empty);
		Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.None)));
		Assert.That(method.ToString(), Is.EqualTo("TestPackage.MockRunTypeLines.Run"));
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
	public const string LetNumber = "	let number = 5";
	public const string LetOther = "	let other = 3";
	public static readonly string[] NestedMethodLines =
	{
		"IsBlaFive returns Boolean",
		LetNumber,
		"	if bla is 5",
		"		return true",
		"	return false"
	};
}