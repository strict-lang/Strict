using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ToTests
{
	[SetUp]
	public void CreateExecutor() => executor = new Executor(TestBehavior.Disabled);

	private Executor executor = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void EvaluateToTextAndNumber()
	{
		using var t = CreateType(nameof(EvaluateToTextAndNumber), "has number", "GetText Text",
			"\tnumber to Text", "GetNumber Number", "\tnumber to Text to Number");
		var instance = new ValueInstance(t, 5);
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "GetText"), instance, []).Value,
			Is.EqualTo("5"));
		Assert.That(
			Convert.ToDouble(executor.
				Execute(t.Methods.Single(m => m.Name == "GetNumber"), instance, []).Value),
			Is.EqualTo(5));
	}

	[Test]
	public void ToCharacterComparison()
	{
		using var t = CreateType(nameof(ToCharacterComparison), "has number", "Compare",
			"\t5 to Character is \"5\"");
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Compare"), null, []).Value,
			Is.EqualTo(true));
	}

	[Test]
	public void ConvertCharacterToNumberAndMultiply()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(ConvertCharacterToNumberAndMultiply), "has character",
					"Convert(number)", "\tcharacter to Number * 10 ^ number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(
			// ReSharper disable once ConfusingCharAsIntegerInConstructor
			executor.Execute(type.Methods[0], new ValueInstance(type, '5'),
				[new ValueInstance(type.GetType(Base.Number), 3)]).Value, Is.EqualTo(5 * 1000));
	}
}
