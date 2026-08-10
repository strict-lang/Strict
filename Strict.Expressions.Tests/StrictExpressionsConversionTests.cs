namespace Strict.Expressions.Tests;

/// <summary>
/// Guards for converting Strict.Expressions C# types to .strict files under Expressions/.
/// </summary>
public sealed class StrictExpressionsConversionTests
{
	[Test]
	public void ExpressionsPackageHasAllExpectedTypes()
	{
		var path = GetExpressionsPath();
		var expected = new[]
		{
			"Expression", "Value", "BooleanExpression", "NumberExpression", "TextExpression",
			"NotExpression", "Return", "VariableCall", "ParameterCall", "Instance", "MemberCall",
			"MethodCall", "Binary", "Declaration", "MutableReassignment", "ListExpression",
			"ListCall", "DictionaryExpression", "IfExpression", "ForExpression", "SelectorIf",
			"To", "TypeComparison", "TypePattern", "ValueInstance", "ValueListInstance",
			"ValueTypeInstance", "ValueDictionaryInstance", "PhraseTokenizer", "ShuntingYard",
			"ExpressionParser", "NumberChars"
		};
		foreach (var typeName in expected)
			Assert.That(File.Exists(Path.Combine(path, typeName + ".strict")), Is.True,
				typeName + ".strict must exist");
	}

	[Test]
	public void ExpressionsStrictFilesAvoidLanguageLegacyFieldNames()
	{
		// Forbidden Language-era stringly fields (see StrictLanguageConversionTests).
		// AST types may store type names as Text metadata (returnTypeName, memberTypeName, etc.).
		var forbidden = new[] { "elementName", "expressionText", "typeNames" };
		var offenders = Directory.GetFiles(GetExpressionsPath(), "*.strict").
			SelectMany(file => File.ReadAllLines(file).Select((line, lineIndex) =>
				new { file, line, LineNumber = lineIndex + 1 })).
			Where(item => item.line.StartsWith("has ", StringComparison.Ordinal) &&
				forbidden.Any(name => item.line.Contains(name, StringComparison.Ordinal))).
			Select(item => Path.GetFileName(item.file) + ":" + item.LineNumber + ": " + item.line.Trim());
		Assert.That(offenders, Is.Empty);
	}

	[Test]
	public async Task LoadExpressionBaseFromExpressionsPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Expressions");
		var expression = package.GetType("Expression");
		Assert.That(expression.Members.Count, Is.GreaterThanOrEqualTo(3));
		Assert.That(expression.FindMember("resultTypeName"), Is.Not.Null);
		Assert.That(expression.FindMember("line"), Is.Not.Null);
	}

	[Test]
	public async Task LoadExpressionParserFromExpressionsPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Expressions");
		var parser = package.GetType("ExpressionParser");
		Assert.That(parser.Methods.Any(method => method.Name == "ParseLine"), Is.True);
		Assert.That(parser.Methods.Any(method => method.Name == "IsBinary"), Is.True);
	}

	[Test]
	public async Task LoadShuntingYardFromExpressionsPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Expressions");
		var shuntingYard = package.GetType("ShuntingYard");
		Assert.That(shuntingYard.Methods.Any(method => method.Name == "Postfix"), Is.True);
		Assert.That(shuntingYard.Methods.Any(method => method.Name == "OperatorPrecedence"), Is.True);
	}

	[Test]
	public async Task LoadValueAndLiteralsFromExpressionsPackage()
	{
		using var package =
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage("Strict/Expressions");
		Assert.That(package.GetType("Value").Members.Any(member => member.Name == "data"), Is.True);
		Assert.That(package.GetType("BooleanExpression").Methods.Any(method => method.Name == "Parse"),
			Is.True);
		Assert.That(package.GetType("NumberExpression").Methods.Any(method => method.Name == "Parse"),
			Is.True);
		Assert.That(package.GetType("TextExpression").Methods.Any(method => method.Name == "Parse"),
			Is.True);
	}

	private static string GetExpressionsPath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Expressions");
}
