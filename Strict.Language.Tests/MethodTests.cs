using NUnit.Framework;

namespace Strict.Language.Tests;

public class MethodTests
{
	[SetUp]
	public void CreateType() =>
		type = new Type(new TestPackage(), nameof(TypeTests), null!).Parse(@"has log
Run
	log.WriteLine");

	private Type type = null!;

	[Test]
	public void MustMustHaveAName() =>
		Assert.Throws<Method.InvalidSyntax>(() => new Method(type, null!, new[] { "a b" }));

	[Test]
	public void ParametersMustNotBeEmpty() =>
		Assert.Throws<Method.EmptyParametersMustBeRemoved>(() =>
			new Method(type, null!, new[] { "a()" }));

	[Test]
	public void ParseDefinition()
	{
		var method = new Method(type, null!, new[] { Run });
		Assert.That(method.Name, Is.EqualTo(Run));
		Assert.That(method.Parameters, Is.Empty);
		Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.None)));
	}

	[Test]
	public void ParseFrom()
	{
		var method = new Method(type, null!, new[] { "from(number)" });
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
		"IsBla5 returns Boolean", LetNumber, "	if bla is 5", "		return true", "	return false"
	};
}