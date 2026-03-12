using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ToTests
{
	[SetUp]
	public void CreateExecutor() =>
		interpreter = new Interpreter(TestPackage.Instance, TestBehavior.Disabled);

	private Interpreter interpreter = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void EvaluateToTextAndNumber()
	{
		using var t = CreateType(nameof(EvaluateToTextAndNumber), "has number", "GetText Text",
			"\tnumber to Text", "GetNumber Number", "\tnumber to Text to Number");
		var instance = new ValueInstance(t, 5);
		Assert.That(interpreter.Execute(t.Methods.Single(m => m.Name == "GetText"), instance, []).Text,
			Is.EqualTo("5"));
		Assert.That(interpreter.Execute(t.Methods.Single(m => m.Name == "GetNumber"), instance, []).Number,
			Is.EqualTo(5));
	}

	[Test]
	public void ToCharacterComparison()
	{
		using var t = CreateType(nameof(ToCharacterComparison), "has number", "Compare",
			"\t5 to Character is \"5\"");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Compare"), interpreter.noneInstance, []).Boolean,
			Is.EqualTo(true));
	}

	[Test]
	public void ConvertCharacterToNumberAndMultiply()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(ConvertCharacterToNumberAndMultiply), "has character",
					"Convert(number)", "\tcharacter to Number * 10 ^ number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(interpreter.Execute(type.Methods[0], new ValueInstance(type, '5'),
			[new ValueInstance(type.GetType(Type.Number), 3)]).Number, Is.EqualTo(5 * 1000));
	}

	[Test]
	public void ComplexTypeToTextDisplaysMemberValues()
	{
		using var pointType = CreateType(nameof(ComplexTypeToTextDisplaysMemberValues) + "Point",
			"has x Number", "has y Number");
		using var t = CreateType(nameof(ComplexTypeToTextDisplaysMemberValues),
			"has point " + nameof(ComplexTypeToTextDisplaysMemberValues) + "Point",
			"GetText Text", "\tpoint to Text");
		var pointInstance = new ValueInstance(pointType, [
			new ValueInstance(interpreter.numberType, 10),
			new ValueInstance(interpreter.numberType, 20)
		]);
		var typeInstance = new ValueInstance(t, [pointInstance]);
		Assert.That(interpreter.Execute(t.Methods.Single(m => m.Name == "GetText"), typeInstance, []).Text,
			Is.EqualTo("(10, 20)"));
	}

	[Test]
	public void NumberPlusTextConcatenatesCorrectly()
	{
		using var t = CreateType(nameof(NumberPlusTextConcatenatesCorrectly),
			"has number", "GetText Text", "\tnumber + \" items\"");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "GetText"), new ValueInstance(t, 5), []).Text,
			Is.EqualTo("5 items"));
	}
}