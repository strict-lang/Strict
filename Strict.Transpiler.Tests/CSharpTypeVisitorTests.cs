using NUnit.Framework;
using Strict.Transpiler.Roslyn;
using Strict.Language;
using Strict.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Transpiler.Tests;

public sealed class CSharpTypeVisitorTests : TestCSharpGenerator
{
	[Test]
	public void GenerateHelloWorldApp()
	{
		using var program = CreateHelloWorldProgramType();
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic static void Main()" + Environment.NewLine + "\t{"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\t\tConsole.WriteLine(\"Hello World\");"));
	}

	private static void AssertProgramClass(CSharpTypeVisitor visitor)
	{
		Assert.That(visitor.Name, Is.EqualTo("Program"));
		Assert.That(visitor.FileContent, Contains.Substring("public class Program"));
		Assert.That(visitor.FileContent.EndsWith("}", StringComparison.InvariantCulture), Is.True,
			visitor.FileContent);
	}

	[Test]
	public void GenerateAppWithImplementingAnotherType()
	{
		using var _ = new Type(package,
				new TypeLines("BaseProgram", "Run")).
			ParseMembersAndMethods(parser);
		using var program = new Type(package,
			new TypeLines("DerivedProgram", "has BaseProgram", "has logger", "Run",
				"\tlogger.Log(\"Hello World\")")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		Assert.That(visitor.Name, Is.EqualTo("DerivedProgram"));
		Assert.That(visitor.FileContent, Contains.Substring("public class DerivedProgram"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic static void Main()" + Environment.NewLine + "\t{"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\t\tConsole.WriteLine(\"Hello World\");"));
	}

	[TestCase("number", "int")]
	[TestCase("boolean", "bool")]
	[TestCase("file", "FileStream")]
	public void GenerateInterface(string parameter, string expectedType)
	{
		using var interfaceType =
			new Type(package, new TypeLines(Computer, $"Compute({parameter})")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("public interface " + Computer));
		Assert.That(visitor.FileContent,
			Contains.Substring($"\tvoid Compute({expectedType} {parameter});" + Environment.NewLine));
	}

	private const string Computer = "Computer";

	[Test]
	public void GenerateTypeThatImplementsMultipleTraits()
	{
		using var program = new Type(package, new TypeLines(
				// @formatter.off
				"Program",
				"has textReader",
				"has system",
				"ReadLines Texts",
				"\tsystem.Write(\"implementing system trait\")",
				"\tReadLines is \"ReadLines successfully\"",
				"\t\"ReadLines successfully\"",
				"Write(lines Texts)",
				"\tconstant stringBuilder = \"printed successfully\"",
				"\tfor lines",
				"\t\tsystem.Write(value)")).
			// @formatter.on
			ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent, Contains.Substring("\tpublic List<string> ReadLines()"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\t\tConsole.WriteLine(\"implementing system trait\");"));
		Assert.That(visitor.FileContent, Contains.Substring("\tpublic void Write(List<string> lines)"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\t\tvar stringBuilder = \"printed successfully\";"));
		Assert.That(visitor.FileContent, Contains.Substring("\t\tforeach (var value in lines)"));
		Assert.That(visitor.FileContent, Contains.Substring("\t\t\tConsole.WriteLine(value);"));
	}

	[Test]
	public void Import()
	{
		var interfaceType =
			new Type(package,
					new TypeLines(Computer, "has inputValue = 5", "has logger", "Run", "\tlogger.Log(inputValue)")).
				ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("namespace " + package.Name + ";"));
		Assert.That(visitor.FileContent, Contains.Substring("public class " + Computer));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic static void Main()" + Environment.NewLine));
	}

	[Test]
	public void MemberInitializer()
	{
		var program =
			new Type(package,
				new TypeLines(Computer, "has number", "has file = \"test.txt\"", "Run",
					"\tfile.Write(number to Text)")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("public class " + Computer));
		Assert.That(visitor.FileContent, Contains.Substring("\tprivate int number"));
		Assert.That(visitor.FileContent,
			Contains.Substring(
				"\tprivate static FileStream file = new FileStream(\"test.txt\", FileMode.OpenOrCreate);"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic static void Main()" + Environment.NewLine));
	}

	[Test]
	public void LocalMemberNotFound() =>
		Assert.That(
			() => new CSharpTypeVisitor(
				new Type(package,
					new TypeLines(Computer, "has logger", "Run", "\tconstant random = logger.unknown")).ParseMembersAndMethods(parser)),
			Throws.InstanceOf<MethodExpressionParser.MemberOrMethodNotFound>());

	[Test]
	public void AccessLocalVariableAfterDeclaration() =>
		Assert.That(
			new CSharpTypeVisitor(
				new Type(package,
					new TypeLines(Computer, "has logger", "has file", "Run", "\tconstant random = \"test\"",
						"\tlogger.Log(random)")).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring("\tConsole.WriteLine(random);"));

	[TestCase("\tvar file = new FileStream(\"test.txt\", FileMode.OpenOrCreate);",
		"has number", "Run", "\tconstant file = File(\"test.txt\")","\tfile.Write(number to Text)")]
	[TestCase("\tnew FileStream(\"test\", FileMode.OpenOrCreate).Write(number.ToString());",
		"has number", "Run", "\tFile(\"test\").Write(number to Text)")]
	public void InitializeValueUsingConstructorInsideMethod(string expected, params string[] code) =>
		Assert.That(new CSharpTypeVisitor(new Type(package, new TypeLines(Computer, code)).
			ParseMembersAndMethods(parser)).FileContent, Contains.Substring(expected));

	[TestCase("ll + mm", "ll + mm")]
	[TestCase("ll - mm", "ll - mm")]
	[TestCase("ll * mm", "ll * mm")]
	public void ListsBinaryOperation(string code, string expected) =>
		Assert.That(new CSharpTypeVisitor(new Type(package, new TypeLines(Computer,
			//@formatter:off
			"has logger",
			"Run",
			"\tconstant ll = (1, 2) + (3, 4)",
			"\tconstant mm = (5, 6)",
			"\tconstant rr = " + code)).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring($"\tvar rr = {expected};"));
				//@formatter:on

	[Test]
	public void GenerateListTypeProgram()
	{
		var program =
			new Type(TestPackage.Instance,
				new TypeLines("Program", "has numbers", "TestListsMethod Numbers",
					"\t(1, 2, 3) + 5", "\tnumbers")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent, Contains.Substring(@"	private List<int> numbers"));
	}

	[Test]
	public void GenerateNestedBodyProgram()
	{
		var program = new Type(package, new TypeLines(
				// @formatter.off
				"Program", "has system", "NestedMethod Number", "	NestedMethod is 5", "	if 5 is 5",
				"		if 5 is not 6", "			constant aa = 5", "		else", "			constant bb = 5")).
			// @formatter.on
			ParseMembersAndMethods(parser);
		var fileContent = new CSharpTypeVisitor(program).FileContent;
		Assert.That(fileContent, Contains.Substring("namespace SourceGeneratorTests;"));
		Assert.That(fileContent, Contains.Substring("public class Program"));
		Assert.That(fileContent, Contains.Substring("\tprivate System system;"));
		Assert.That(fileContent, Contains.Substring("\tpublic int NestedMethod()"));
		Assert.That(fileContent, Contains.Substring("\t\tNestedMethod() == 5;"));
		Assert.That(fileContent, Contains.Substring("\t\tif (5 == 5)"));
		Assert.That(fileContent, Contains.Substring("\t\t\tif (5 is not 6)"));
		Assert.That(fileContent, Contains.Substring("\t\t\t\tvar aa = 5;"));
		Assert.That(fileContent, Contains.Substring("\t\t\telse"));
		Assert.That(fileContent, Contains.Substring("\t\t\t\tvar bb = 5;"));
	}
}
