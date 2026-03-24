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
	public void EvaluateListOfTextsToNumbers()
	{
		using var type = CreateType(nameof(EvaluateListOfTextsToNumbers), "has number", "Run Numbers",
			"\t(\"1\", \"2\") to Numbers");
		Assert.That(
			interpreter.Execute(type.Methods.Single(method => method.Name == Method.Run),
				interpreter.noneInstance, []).List.Items,
			Is.EqualTo(new[]
			{
				new ValueInstance(type.GetType(Type.Number), 1),
				new ValueInstance(type.GetType(Type.Number), 2)
			}));
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
	public void UpperConvertsTextToUppercase()
	{
		using var type = CreateType(nameof(UpperConvertsTextToUppercase), "has number", "Run Text",
			"\t\"HeLlo\".Upper");
		Assert.That(interpreter.Execute(type.Methods.Single(method => method.Name == Method.Run),
			interpreter.noneInstance, []).Text, Is.EqualTo("HELLO"));
	}

	[Test]
	public void LowerConvertsTextToLowercase()
	{
		using var type = CreateType(nameof(LowerConvertsTextToLowercase), "has number", "Run Text",
			"\t\"HeLlo\".Lower");
		Assert.That(interpreter.Execute(type.Methods.Single(method => method.Name == Method.Run),
			interpreter.noneInstance, []).Text, Is.EqualTo("hello"));
	}

	[Test]
	public void TextCharactersLengthMatchesSourceTextLength()
	{
		using var type = CreateType(nameof(TextCharactersLengthMatchesSourceTextLength), "has number",
			"Run Number", "\t\"hello\".characters.Length");
		Assert.That(interpreter.Execute(type.Methods.Single(method => method.Name == Method.Run),
			interpreter.noneInstance, []).Number, Is.EqualTo(5));
	}

	[TestCase("hello", "l", 3)]
	[TestCase("hello", "x", -1)]
	public void LastIndexOfFindsLastMatchOrMissingValue(string text, string searchText,
		double expected)
	{
		using var type = CreateType(nameof(LastIndexOfFindsLastMatchOrMissingValue), "has number",
			"Run Number", "\t\"" + text + "\".LastIndexOf(\"" + searchText + "\")");
		Assert.That(interpreter.Execute(type.Methods.Single(method => method.Name == Method.Run),
			interpreter.noneInstance, []).Number, Is.EqualTo(expected));
	}

	[TestCase("hello", "hel", true)]
	[TestCase("yo mama", "mama", false)]
	[TestCase("hello", "hello", true)]
	public void StartsWithReturnsTrueWhenTextStartsWithPrefix(string text, string prefix,
		bool expected)
	{
		using var type = CreateType(nameof(StartsWithReturnsTrueWhenTextStartsWithPrefix), "has number",
			"Run Boolean", "\t\"" + text + "\".StartsWith(\"" + prefix + "\")");
		Assert.That(interpreter.Execute(type.Methods.Single(method => method.Name == Method.Run),
			interpreter.noneInstance, []).Boolean, Is.EqualTo(expected));
	}
}