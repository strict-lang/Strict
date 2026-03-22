using static Strict.Language.Member;

namespace Strict.Language.Tests;

/// <summary>
/// Tests for converting Strict.Language C# types to .strict files.
/// See the Language/ directory at the repository root for the .strict implementations.
/// Just as C# uses workarounds like Type.ValueLowercase for the "value" keyword or
/// 'using Type = Strict.Language.Type' to avoid System.Type conflicts, Strict uses the
/// same approach: rename conflicting constants (e.g. MutableKeyword, KindBoolean) rather
/// than treating conflicts as blockers.
/// </summary>
public sealed class StrictLanguageConversionTests
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private ExpressionParser parser = null!;

	[Test]
	public void LimitTypeHasCorrectConstants()
	{
		using var limitType = new Type(TestPackage.Instance,
			new TypeLines("Limit",
				"constant MethodLength = 12",
				"constant ParameterCount = 4",
				"constant MethodCount = 15",
				"constant LineCount = 256",
				"constant NestingLevel = 5",
				"constant CharacterCount = 120",
				"constant MultiLineCharacterCount = 100",
				"constant MemberCount = 15",
				"constant MemberCountForEnums = 40",
				"constant NameMaxLimit = 50",
				"constant NameMinLimit = 2")).ParseMembersAndMethods(parser);
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
		Assert.That(limitType.IsEnum, Is.True);
		Assert.That(limitType.Members[0].Name, Is.EqualTo("MethodLength"));
		Assert.That(limitType.Members[0].Type.Name, Is.EqualTo(Type.Number));
		Assert.That(limitType.Members[0].IsConstant, Is.True);
		Assert.That(limitType.Members[^1].Name, Is.EqualTo("NameMinLimit"));
	}

	[Test]
	public void LoadLimitTypeFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var limitLines = new TypeLines("Limit",
			File.ReadAllLines(Path.Combine(langPath, "Limit.strict")));
		using var limitType = new Type(TestPackage.Instance, limitLines).ParseMembersAndMethods(parser);
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
		Assert.That(limitType.IsEnum, Is.True);
	}

	/// <summary>
	/// Keyword.cs has string constants like Has="has", Mutable="mutable", etc.
	/// Most uppercase names (Has, Constant, Let, If, Else, For, With, Return) work as-is.
	/// The one conflict: "Mutable" is a built-in Strict type, so we name it MutableKeyword,
	/// following the same convention as C#'s Type.ValueLowercase for "value".
	/// </summary>
	[Test]
	public void KeywordTypeHasCorrectTextConstants()
	{
		using var keywordType = new Type(TestPackage.Instance,
			new TypeLines("Keyword",
				"constant Has = \"has\"",
				"constant Constant = \"constant\"",
				"constant Let = \"let\"",
				"constant MutableKeyword = \"mutable\"",
				"constant If = \"if\"",
				"constant Else = \"else\"",
				"constant For = \"for\"",
				"constant With = \"with\"",
				"constant Return = \"return\"")).ParseMembersAndMethods(parser);
		Assert.That(keywordType.Members.Count, Is.EqualTo(9));
		Assert.That(keywordType.Members[0].Name, Is.EqualTo("Has"));
		Assert.That(keywordType.Members[0].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(keywordType.Members[3].Name, Is.EqualTo("MutableKeyword"));
		Assert.That(keywordType.Members[3].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(keywordType.Members[3].IsConstant, Is.True);
	}

	[Test]
	public void LoadKeywordTypeFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("Keyword", File.ReadAllLines(Path.Combine(langPath, "Keyword.strict")));
		using var keywordType = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(keywordType.Members.Count, Is.EqualTo(9));
		Assert.That(keywordType.IsEnum, Is.True);
	}

	/// <summary>
	/// TypeKind.cs is an enum with values None, Boolean, Number, etc. In Strict, those names
	/// conflict with built-in types (same issue as C# with System.Type vs Strict.Language.Type).
	/// The workaround: prefix with "Kind" (KindBoolean, KindNumber, etc.) — no conflict, still clear.
	/// </summary>
	[Test]
	public void TypeKindConstantsUseKindPrefixToAvoidBuiltInTypeNameConflicts()
	{
		using var typeKind = new Type(TestPackage.Instance,
			new TypeLines("TypeKind",
				"constant KindNone",
				"constant KindBoolean",
				"constant KindNumber",
				"constant KindText",
				"constant KindCharacter",
				"constant KindList",
				"constant KindDictionary",
				"constant KindError",
				"constant KindEnum",
				"constant KindIterator",
				"constant KindAny",
				"constant KindUnknown")).ParseMembersAndMethods(parser);
		Assert.That(typeKind.IsEnum, Is.True);
		Assert.That(typeKind.Members.Count, Is.EqualTo(12));
		Assert.That(typeKind.Members[0].Name, Is.EqualTo("KindNone"));
		Assert.That(typeKind.Members[0].Type.Name, Is.EqualTo(Type.Number));
		Assert.That(typeKind.Members[1].Name, Is.EqualTo("KindBoolean"));
	}

	[Test]
	public void LoadTypeKindFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("TypeKind",
			File.ReadAllLines(Path.Combine(langPath, "TypeKind.strict")));
		using var typeKind = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(typeKind.Members.Count, Is.EqualTo(12));
		Assert.That(typeKind.IsEnum, Is.True);
	}

	[Test]
	public void UnaryOperatorHasNotConstant()
	{
		using var unaryOp = new Type(TestPackage.Instance,
			new TypeLines("UnaryOperator",
				"constant Not = \"not\"")).ParseMembersAndMethods(parser);
		Assert.That(unaryOp.Members.Count, Is.EqualTo(1));
		Assert.That(unaryOp.Members[0].Name, Is.EqualTo("Not"));
		Assert.That(unaryOp.Members[0].Type.Name, Is.EqualTo(Type.Text));
	}

	[Test]
	public void LoadUnaryOperatorFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("UnaryOperator",
			File.ReadAllLines(Path.Combine(langPath, "UnaryOperator.strict")));
		using var unaryOp = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(unaryOp.Members.Count, Is.EqualTo(1));
		Assert.That(unaryOp.Members[0].Name, Is.EqualTo("Not"));
	}

	/// <summary>
	/// BinaryOperator.cs has 16 operator string constants like Plus="+", Is="is", etc.
	/// None of these names conflict with Strict type names or keywords — all map directly.
	/// </summary>
	[Test]
	public void BinaryOperatorTypeHasCorrectTextConstants()
	{
		using var binaryOp = new Type(TestPackage.Instance,
			new TypeLines("BinaryOperator",
				"constant Plus = \"+\"",
				"constant Minus = \"-\"",
				"constant Multiply = \"*\"",
				"constant Divide = \"/\"",
				"constant Power = \"^\"",
				"constant Modulate = \"%\"",
				"constant Smaller = \"<\"",
				"constant Greater = \">\"",
				"constant SmallerOrEqual = \"<=\"",
				"constant GreaterOrEqual = \">=\"",
				"constant Is = \"is\"",
				"constant In = \"in\"",
				"constant To = \"to\"",
				"constant And = \"and\"",
				"constant Or = \"or\"",
				"constant Xor = \"xor\"")).ParseMembersAndMethods(parser);
		Assert.That(binaryOp.Members.Count, Is.EqualTo(16));
		Assert.That(binaryOp.Members[0].Name, Is.EqualTo("Plus"));
		Assert.That(binaryOp.Members[0].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(binaryOp.Members[0].IsConstant, Is.True);
		Assert.That(binaryOp.Members[^1].Name, Is.EqualTo("Xor"));
	}

	[Test]
	public void LoadBinaryOperatorFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("BinaryOperator",
			File.ReadAllLines(Path.Combine(langPath, "BinaryOperator.strict")));
		using var binaryOp = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(binaryOp.Members.Count, Is.EqualTo(16));
		Assert.That(binaryOp.IsEnum, Is.True);
	}

	/// <summary>
	/// TypeLines.cs is a data container holding a type's name and source lines.
	/// In Strict it becomes a plain type with two members and a to Text method returning the name.
	/// Strict's naming rule: member name must not match a different type — so we use "typeName"
	/// (no "TypeName" type exists) instead of "name" (which conflicts with the Name type).
	/// The DependentTypes computation (string parsing) is deferred to a later stage.
	/// </summary>
	[Test]
	public void TypeLinesHasTwoMembersAndToTextMethod()
	{
		using var typeLines = new Type(TestPackage.Instance,
			new TypeLines("TypeLines",
				"has typeName Text",
				"has lines Texts",
				"to Text",
				"\ttypeName")).ParseMembersAndMethods(parser);
		Assert.That(typeLines.Members.Count, Is.EqualTo(2));
		Assert.That(typeLines.Members[0].Name, Is.EqualTo("typeName"));
		Assert.That(typeLines.Members[0].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(typeLines.Members[1].Name, Is.EqualTo("lines"));
		Assert.That(typeLines.Members[1].Type.Name, Does.StartWith(Type.List));
		Assert.That(typeLines.Methods.Count, Is.EqualTo(1));
		Assert.That(typeLines.Methods[0].Name, Is.EqualTo("to"));
		Assert.That(typeLines.Methods[0].ReturnType.Name, Is.EqualTo(Type.Text));
		Assert.That(typeLines.Methods[0].GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadTypeLinesFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("TypeLines",
			File.ReadAllLines(Path.Combine(langPath, "TypeLines.strict")));
		using var typeLines = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(typeLines.Members.Count, Is.EqualTo(2));
		Assert.That(typeLines.Members[0].Name, Is.EqualTo("typeName"));
		Assert.That(typeLines.Members[1].Name, Is.EqualTo("lines"));
		Assert.That(typeLines.Methods.Count, Is.EqualTo(3));
		Assert.That(typeLines.Methods.Any(method => method.Name == "to"), Is.True);
		Assert.That(typeLines.Methods.Any(method => method.Name == "CountMembers"), Is.True);
		Assert.That(typeLines.Methods.Any(method => method.Name == "MemberLines"), Is.True);
		Assert.That(typeLines.Methods.Single(method => method.Name == "CountMembers").ReturnType.Name,
			Is.EqualTo(Type.Number));
		Assert.That(typeLines.Methods.Single(method => method.Name == "MemberLines").ReturnType.Name,
			Does.StartWith(Type.List));
		foreach (var method in typeLines.Methods)
			Assert.That(method.GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
	}

	/// <summary>
	/// NamedType.cs is the abstract base for Parameter, Member, and Variable — it holds a name
	/// and a type reference and provides a canonical to Text. In Strict we model it as a concrete
	/// type (no abstract classes) with two Text members: the element name and the type name.
	/// Strict naming rule: "name" → conflicts with Name type, so use "elementName".
	/// The to Text method mirrors C# ToString: Name + " " + Type.
	/// The inline test exercises the method body and verifies string concatenation works.
	/// </summary>
	[Test]
	public void NamedTypeToTextConcatenatesElementAndTypeName()
	{
		using var namedType = new Type(TestPackage.Instance,
			new TypeLines("NamedType",
				"has elementName Text",
				"has typeName Text",
				"to Text",
				"\tNamedType(\"count\", \"Number\") to Text is \"count Number\"",
				"\telementName + \" \" + typeName")).ParseMembersAndMethods(parser);
		Assert.That(namedType.Members.Count, Is.EqualTo(2));
		Assert.That(namedType.Members[0].Name, Is.EqualTo("elementName"));
		Assert.That(namedType.Members[0].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(namedType.Members[1].Name, Is.EqualTo("typeName"));
		Assert.That(namedType.Members[1].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(namedType.Methods.Count, Is.EqualTo(1));
		Assert.That(namedType.Methods[0].Name, Is.EqualTo("to"));
		Assert.That(namedType.Methods[0].ReturnType.Name, Is.EqualTo(Type.Text));
		Assert.That(namedType.Methods[0].GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadNamedTypeFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		var lines = new TypeLines("NamedType",
			File.ReadAllLines(Path.Combine(langPath, "NamedType.strict")));
		using var namedType = new Type(TestPackage.Instance, lines).ParseMembersAndMethods(parser);
		Assert.That(namedType.Members.Count, Is.EqualTo(2));
		Assert.That(namedType.Members[0].Name, Is.EqualTo("elementName"));
		Assert.That(namedType.Members[1].Name, Is.EqualTo("typeName"));
		Assert.That(namedType.Methods.Count, Is.EqualTo(1));
		Assert.That(namedType.Methods[0].GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadParameterFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var variableLines = new TypeLines("Variable",
			File.ReadAllLines(Path.Combine(langPath, "Variable.strict")));
		using var _ = new Type(conversionPackage, variableLines).ParseMembersAndMethods(parser);
		var lines = new TypeLines("LanguageParameter",
			File.ReadAllLines(Path.Combine(langPath, "Parameter.strict")));
		using var parameterType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(parameterType.Members.Count, Is.EqualTo(1));
		Assert.That(parameterType.Members[0].Type.Name, Is.EqualTo("Variable"));
		Assert.That(parameterType.Methods.Count, Is.EqualTo(1));
		Assert.That(parameterType.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public void LoadMemberFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var variableLines = new TypeLines("Variable",
			File.ReadAllLines(Path.Combine(langPath, "Variable.strict")));
		using var _ = new Type(conversionPackage, variableLines).ParseMembersAndMethods(parser);
		var lines = new TypeLines("LanguageMember",
			File.ReadAllLines(Path.Combine(langPath, "Member.strict")));
		using var memberType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(memberType.Members.Count, Is.EqualTo(2));
		Assert.That(memberType.Members[0].Type.Name, Is.EqualTo("Variable"));
		Assert.That(memberType.Members[1].Name, Is.EqualTo("isConstant"));
		Assert.That(memberType.Methods.Count, Is.EqualTo(1));
		Assert.That(memberType.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public void LoadVariableFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var lines = new TypeLines("LanguageVariable",
			File.ReadAllLines(Path.Combine(langPath, "Variable.strict")));
		using var variableType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(variableType.Members.Count, Is.EqualTo(4));
		Assert.That(variableType.Members[0].Name, Is.EqualTo("elementName"));
		Assert.That(variableType.Members[1].Name, Is.EqualTo("typeName"));
		Assert.That(variableType.Members[2].Name, Is.EqualTo("isMutable"));
		Assert.That(variableType.Members[3].Name, Is.EqualTo("initialValueExpression"));
		Assert.That(variableType.Methods.Count, Is.EqualTo(1));
		Assert.That(variableType.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public void LoadMethodFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var lines = new TypeLines("LanguageMethod",
			File.ReadAllLines(Path.Combine(langPath, "Method.strict")));
		using var methodType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(methodType.Members.Count, Is.EqualTo(5));
		Assert.That(methodType.Members[0].Name, Is.EqualTo("methodName"));
		Assert.That(methodType.Members[1].Name, Is.EqualTo("returnTypeName"));
		Assert.That(methodType.Members[2].Name, Is.EqualTo("parameterNames"));
		Assert.That(methodType.Members[3].Name, Is.EqualTo("isPublic"));
		Assert.That(methodType.Members[4].Name, Is.EqualTo("isTrait"));
		Assert.That(methodType.Methods.Count, Is.EqualTo(1));
		Assert.That(methodType.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public void LoadContextFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var lines = new TypeLines("LanguageContext",
			File.ReadAllLines(Path.Combine(langPath, "Context.strict")));
		using var contextType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(contextType.Members.Count, Is.EqualTo(1));
		Assert.That(contextType.Members[0].Name, Is.EqualTo("contextName"));
		Assert.That(contextType.Methods.Count, Is.EqualTo(3));
		Assert.That(contextType.Methods.Any(method => method.Name == "FindType"), Is.True);
		Assert.That(contextType.Methods.Any(method => method.Name == "TryGetType"), Is.True);
		Assert.That(contextType.Methods.Any(method => method.Name == "GetType"), Is.True);
	}

	[Test]
	public void LoadPackageFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var lines = new TypeLines("Package",
			File.ReadAllLines(Path.Combine(langPath, "Package.strict")));
		using var packageType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(packageType.Members.Count, Is.EqualTo(4));
		Assert.That(packageType.Members[0].Name, Is.EqualTo("packageName"));
		Assert.That(packageType.Members[1].Name, Is.EqualTo("fullName"));
		Assert.That(packageType.Members[2].Name, Is.EqualTo("typeNames"));
		Assert.That(packageType.Members[3].Name, Is.EqualTo("parentName"));
		Assert.That(packageType.Methods.Count, Is.EqualTo(2));
		Assert.That(packageType.Methods.Any(method => method.Name == "FindType"), Is.True);
		Assert.That(packageType.Methods.Any(method => method.Name == "AddType"), Is.True);
		Assert.That(packageType.Methods.Single(method => method.Name == "AddType").ReturnType.Name,
			Is.EqualTo("Mutable(Package)"));
	}

	[Test]
	public void LoadTypeFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var lines = new TypeLines("LanguageType",
			File.ReadAllLines(Path.Combine(langPath, "Type.strict")));
		using var typeType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(typeType.Members.Count, Is.EqualTo(6));
		Assert.That(typeType.Members[0].Name, Is.EqualTo("typeName"));
		Assert.That(typeType.Members[1].Name, Is.EqualTo("packageName"));
		Assert.That(typeType.Members[2].Name, Is.EqualTo("memberNames"));
		Assert.That(typeType.Members[3].Name, Is.EqualTo("methodNames"));
		Assert.That(typeType.Members[4].Name, Is.EqualTo("isTrait"));
		Assert.That(typeType.Members[5].Name, Is.EqualTo("isGeneric"));
		Assert.That(typeType.Methods.Count, Is.EqualTo(1));
		Assert.That(typeType.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public void LoadBodyFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var lines = new TypeLines("LanguageBody",
			File.ReadAllLines(Path.Combine(langPath, "Body.strict")));
		using var bodyType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(bodyType.Members.Count, Is.EqualTo(3));
		Assert.That(bodyType.Members[0].Name, Is.EqualTo("methodName"));
		Assert.That(bodyType.Members[1].Name, Is.EqualTo("expressionTexts"));
		Assert.That(bodyType.Members[2].Name, Is.EqualTo("lineCount"));
		Assert.That(bodyType.Methods.Count, Is.EqualTo(1));
		Assert.That(bodyType.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public void LoadExpressionFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var lines = new TypeLines("LanguageExpression",
			File.ReadAllLines(Path.Combine(langPath, "Expression.strict")));
		using var expressionType = new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(expressionType.Members.Count, Is.EqualTo(3));
		Assert.That(expressionType.Members[0].Name, Is.EqualTo("returnTypeName"));
		Assert.That(expressionType.Members[1].Name, Is.EqualTo("lineNumber"));
		Assert.That(expressionType.Members[2].Name, Is.EqualTo("isMutable"));
		Assert.That(expressionType.Methods.Count, Is.EqualTo(2));
		Assert.That(expressionType.Methods.Any(method => method.Name == "IsConstant"), Is.True);
		Assert.That(expressionType.Methods.Any(method => method.Name == "to"), Is.True);
	}

	[Test]
	public void LoadConcreteExpressionFromLanguageDirectory()
	{
		var langPath = GetLanguagePath();
		using var conversionPackage = new Package(TestPackage.Instance,
			"Conv" + Guid.NewGuid().ToString("N")[..8]);
		var expressionLines = new TypeLines("Expression",
			File.ReadAllLines(Path.Combine(langPath, "Expression.strict")));
		using var _ = new Type(conversionPackage, expressionLines).ParseMembersAndMethods(parser);
		var lines = new TypeLines("LanguageConcreteExpression",
			File.ReadAllLines(Path.Combine(langPath, "ConcreteExpression.strict")));
		using var concreteExpressionType =
			new Type(conversionPackage, lines).ParseMembersAndMethods(parser);
		Assert.That(concreteExpressionType.Members.Count, Is.EqualTo(2));
		Assert.That(concreteExpressionType.Members[0].Type.Name, Is.EqualTo("Expression"));
		Assert.That(concreteExpressionType.Members[1].Name, Is.EqualTo("expressionText"));
		Assert.That(concreteExpressionType.Methods.Count, Is.EqualTo(2));
		Assert.That(concreteExpressionType.Methods.Any(method => method.Name == "IsConstant"), Is.True);
		Assert.That(concreteExpressionType.Methods.Any(method => method.Name == "to"), Is.True);
	}

	private static string GetLanguagePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Language");
}
