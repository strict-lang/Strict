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
		using var limitType = CreateConversionPackageType(TestPackage.Instance, "Limit");
		Assert.That(limitType.Members.Count, Is.EqualTo(11));
		Assert.That(limitType.IsEnum, Is.True);
	}

	private Type CreateConversionPackageType(Package package, string typeName) =>
		new Type(package, new TypeLines(typeName, File.ReadAllLines(
			Path.Combine(GetLanguagePath(), $"{typeName}.strict")))).ParseMembersAndMethods(parser);

	private static string GetLanguagePath() =>
		Path.Combine(Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Language");

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
		using var keywordType = CreateConversionPackageType(TestPackage.Instance, "Keyword");
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
		using var typeKind = CreateConversionPackageType(TestPackage.Instance, "TypeKind");
		Assert.That(typeKind.Members.Count, Is.EqualTo(12));
		Assert.That(typeKind.IsEnum, Is.True);
	}

	[Test]
	public void UnaryOperatorHasNotConstant()
	{
		using var unaryOp = new Type(TestPackage.Instance, new TypeLines("UnaryOperator",
			"constant Not = \"not\"")).ParseMembersAndMethods(parser);
		Assert.That(unaryOp.Members.Count, Is.EqualTo(1));
		Assert.That(unaryOp.Members[0].Name, Is.EqualTo("Not"));
		Assert.That(unaryOp.Members[0].Type.Name, Is.EqualTo(Type.Text));
	}

	[Test]
	public void LoadUnaryOperatorFromLanguageDirectory()
	{
		using var unaryOp = CreateConversionPackageType(TestPackage.Instance, "UnaryOperator");
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
		using var binaryOp = CreateConversionPackageType(TestPackage.Instance, "BinaryOperator");
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
	public void TypeLinesSupportsLoadingAndInspectingSourceLines()
	{
		using var typeLines = new Type(TestPackage.Instance,
			new TypeLines("TypeLines",
				"has typeName Text",
				"has lines Texts",
				"from(code Text, name Text) TypeLines",
				"\tTypeLines(name, code.Replace(\"\\r\", \"\").Split(\"\\n\"))",
				"from(filePath Text) TypeLines",
				"\tTypeLines.from(File(Name(filePath)).Read, filePath)",
				"Line(number) Text",
				"\tlines(number)",
				"Count Number",
				"\tlines.Length",
				"to Text",
				"\ttypeName")).ParseMembersAndMethods(parser);
		Assert.That(typeLines.Members.Count, Is.EqualTo(2));
		Assert.That(typeLines.Members[0].Name, Is.EqualTo("typeName"));
		Assert.That(typeLines.Members[0].Type.Name, Is.EqualTo(Type.Text));
		Assert.That(typeLines.Members[1].Name, Is.EqualTo("lines"));
		Assert.That(typeLines.Members[1].Type.Name, Does.StartWith(Type.List));
		Assert.That(typeLines.Methods.Count, Is.EqualTo(5));
		Assert.That(typeLines.Methods.Any(method => method.Name == "Line"), Is.True);
		Assert.That(typeLines.Methods.Any(method => method.Name == "Count"), Is.True);
		Assert.That(typeLines.Methods.Single(method => method.Name == "to").ReturnType.Name,
			Is.EqualTo(Type.Text));
		Assert.That(typeLines.Methods.Single(method => method.Name == "to").GetBodyAndParseIfNeeded(),
			Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadTypeLinesFromLanguageDirectory()
	{
		using var typeLines = CreateConversionPackageType(TestPackage.Instance, "TypeLines");
		Assert.That(typeLines.Members.Count, Is.EqualTo(2));
		Assert.That(typeLines.Members[0].Name, Is.EqualTo("typeName"));
		Assert.That(typeLines.Members[1].Name, Is.EqualTo("lines"));
		Assert.That(typeLines.Methods.Count, Is.EqualTo(7));
		Assert.That(typeLines.Methods.Any(method => method.Name == "to"), Is.True);
		Assert.That(typeLines.Methods.Any(method => method.Name == "Line"), Is.True);
		Assert.That(typeLines.Methods.Any(method => method.Name == "Count"), Is.True);
		Assert.That(typeLines.Methods.Any(method => method.Name == "CountMembers"), Is.True);
		Assert.That(typeLines.Methods.Any(method => method.Name == "MemberLines"), Is.True);
		Assert.That(typeLines.Methods.Single(method => method.Name == "CountMembers").ReturnType.Name,
			Is.EqualTo(Type.Number));
		Assert.That(typeLines.Methods.Single(method => method.Name == "MemberLines").ReturnType.Name,
			Does.StartWith(Type.List));
		foreach (var method in typeLines.Methods.Where(m =>
			m.Name is "to" or "Line" or "Count" or "CountMembers" or "MemberLines"))
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
		using var namedType = CreateConversionPackageType(TestPackage.Instance, "NamedType");
		Assert.That(namedType.Members.Count, Is.EqualTo(2));
		Assert.That(namedType.Members[0].Name, Is.EqualTo("elementName"));
		Assert.That(namedType.Members[1].Name, Is.EqualTo("typeName"));
		Assert.That(namedType.Methods.Count, Is.EqualTo(1));
		Assert.That(namedType.Methods[0].GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadParameterFromLanguageDirectory()
	{
		using var conversionPackage = new Package(TestPackage.Instance,	ConversionPackageName);
		using var variableType = CreateConversionPackageType(conversionPackage, "Variable");
		using var parameterType = CreateConversionPackageType(conversionPackage, "Parameter");
		Assert.That(parameterType.Members.Count, Is.EqualTo(1));
		Assert.That(parameterType.Members[0].Type.Name, Is.EqualTo("Variable"));
		Assert.That(parameterType.Methods.Count, Is.EqualTo(1));
		Assert.That(parameterType.Methods[0].Name, Is.EqualTo("to"));
	}

	private static string ConversionPackageName =>
		Package.TestLanguageConversion + Guid.NewGuid().ToString("N")[..8];

	[Test]
	public void LoadMemberFromLanguageDirectory()
	{
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var variableType = CreateConversionPackageType(conversionPackage, "Variable");
		using var memberType = CreateConversionPackageType(conversionPackage, "Member");
		Assert.That(memberType.Members.Count, Is.EqualTo(2));
		Assert.That(memberType.Members[0].Type.Name, Is.EqualTo("Variable"));
		Assert.That(memberType.Members[1].Name, Is.EqualTo("isConstant"));
		Assert.That(memberType.Methods.Count, Is.EqualTo(1));
		Assert.That(memberType.Methods[0].Name, Is.EqualTo("to"));
	}

	[Test]
	public void LoadVariableFromLanguageDirectory()
	{
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var variableType = CreateConversionPackageType(conversionPackage, "Variable");
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
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var variableType = CreateConversionPackageType(conversionPackage, "Variable");
		using var parameterType = CreateConversionPackageType(conversionPackage, "Parameter");
		using var methodType = CreateConversionPackageType(conversionPackage, "Method");
		Assert.That(methodType.Members.Count, Is.EqualTo(5));
		Assert.That(methodType.Members[0].Name, Is.EqualTo("methodName"));
		Assert.That(methodType.Members[1].Name, Is.EqualTo("returnTypeName"));
		Assert.That(methodType.Members[2].Name, Is.EqualTo("parameters"));
		Assert.That(methodType.Members[3].Name, Is.EqualTo("bodyLines"));
		Assert.That(methodType.Members[4].Name, Is.EqualTo("isPublic"));
		Assert.That(methodType.Methods.Count, Is.EqualTo(6));
		Assert.That(methodType.Methods.Any(method => method.Name == "ParameterNames"), Is.True);
		Assert.That(methodType.Methods.Any(method => method.Name == "BodyLineCount"), Is.True);
		Assert.That(methodType.Methods.Any(method => method.Name == "IsTrait"), Is.True);
		Assert.That(methodType.Methods.Single(method => method.Name == "to").GetBodyAndParseIfNeeded(),
			Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadContextFromLanguageDirectory()
	{
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var contextType = CreateConversionPackageType(conversionPackage, "Context");
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
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var packageType = CreateConversionPackageType(conversionPackage, "Package");
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
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var variableType = CreateConversionPackageType(conversionPackage, "Variable");
		using var memberType = CreateConversionPackageType(conversionPackage, "Member");
		using var parameterType = CreateConversionPackageType(conversionPackage, "Parameter");
		using var methodType = CreateConversionPackageType(conversionPackage, "Method");
		using var typeType = CreateConversionPackageType(conversionPackage, "Type");
		Assert.That(typeType.Members.Count, Is.EqualTo(6));
		Assert.That(typeType.Members[0].Name, Is.EqualTo("typeName"));
		Assert.That(typeType.Members[1].Name, Is.EqualTo("packageName"));
		Assert.That(typeType.Members[2].Name, Is.EqualTo("members"));
		Assert.That(typeType.Members[3].Name, Is.EqualTo("methods"));
		Assert.That(typeType.Members[4].Name, Is.EqualTo("isTrait"));
		Assert.That(typeType.Members[5].Name, Is.EqualTo("isGeneric"));
		Assert.That(typeType.Methods.Count, Is.EqualTo(7));
		Assert.That(typeType.Methods.Any(method => method.Name == "MemberNames"), Is.True);
		Assert.That(typeType.Methods.Any(method => method.Name == "MethodNames"), Is.True);
		Assert.That(typeType.Methods.Any(method => method.Name == "GetMethod"), Is.True);
		Assert.That(typeType.Methods.Single(method => method.Name == "to").GetBodyAndParseIfNeeded(),
			Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadTypeParserFromLanguageDirectory()
	{
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var variableType = CreateConversionPackageType(conversionPackage, "Variable");
		using var parameterType = CreateConversionPackageType(conversionPackage, "Parameter");
		using var memberType = CreateConversionPackageType(conversionPackage, "Member");
		using var methodType = CreateConversionPackageType(conversionPackage, "Method");
		using var typeLinesType = CreateConversionPackageType(conversionPackage, "TypeLines");
		using var typeType = CreateConversionPackageType(conversionPackage, "Type");
		using var typeParserType = CreateConversionPackageType(conversionPackage, "TypeParser");
		Assert.That(typeParserType.Members.Count, Is.EqualTo(1));
		Assert.That(typeParserType.Members[0].Name, Is.EqualTo("packageName"));
		Assert.That(typeParserType.Methods.Any(method => method.Name == "Parse"), Is.True);
		Assert.That(typeParserType.Methods.Any(method => method.Name == "ParseMember"), Is.True);
		Assert.That(typeParserType.Methods.Any(method => method.Name == "ParseMethod"), Is.True);
		Assert.That(typeParserType.Methods.Single(method => method.Name == "Parse").
			GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
	}

	[Test]
	public void LoadBodyFromLanguageDirectory()
	{
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var bodyType = CreateConversionPackageType(conversionPackage, "Body");
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
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var expressionType = CreateConversionPackageType(conversionPackage, "Expression");
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
		using var conversionPackage = new Package(TestPackage.Instance, ConversionPackageName);
		using var expressionType = CreateConversionPackageType(conversionPackage, "Expression");
		using var concreteExpressionType = CreateConversionPackageType(conversionPackage,
			"ConcreteExpression");
		Assert.That(concreteExpressionType.Members.Count, Is.EqualTo(2));
		Assert.That(concreteExpressionType.Members[0].Type.Name, Is.EqualTo("Expression"));
		Assert.That(concreteExpressionType.Members[1].Name, Is.EqualTo("expressionText"));
		Assert.That(concreteExpressionType.Methods.Count, Is.EqualTo(2));
		Assert.That(concreteExpressionType.Methods.Any(method => method.Name == "IsConstant"), Is.True);
		Assert.That(concreteExpressionType.Methods.Any(method => method.Name == "to"), Is.True);
	}
}
