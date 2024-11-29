using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

public sealed class CSharpTypeVisitorTests : TestCSharpGenerator
{
	[Test]
	public void GenerateHelloWorldApp()
	{
		var program = CreateHelloWorldProgramType();
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
		new Type(package,
				new TypeLines("BaseProgram", "Run")).
			ParseMembersAndMethods(parser);
		var program = new Type(package,
			new TypeLines("DerivedProgram", "has BaseProgram", "has log", "Run",
				"\tlog.Write(\"Hello World\")")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		Assert.That(visitor.Name, Is.EqualTo("DerivedProgram"));
		Assert.That(visitor.FileContent, Contains.Substring("public class DerivedProgram : BaseProgram"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic void Run()" + Environment.NewLine + "\t{"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\t\tConsole.WriteLine(\"Hello World\");"));
	}

	[TestCase("number", "int")]
	[TestCase("boolean", "bool")]
	[TestCase("file", "FileStream")]
	public void GenerateInterface(string parameter, string expectedType)
	{
		var interfaceType =
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
		var program = new Type(package, new TypeLines(
				// @formatter.off
				"Program",
				"has Input",
				"has Output",
				"has system",
				"Read Text",
				"\tsystem.WriteLine(\"implementing system trait\")",
				"\tRead is \"Read successfully\"",
				"\t\"Read successfully\"",
				"Write(generic) Boolean",
				"\tWrite(5) is true",
				"\tconstant stringBuilder = \"printed successfully\"",
				"\ttrue")).
			// @formatter.on
			ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent, Contains.Substring(@"	public string Read()
	{
		Console.WriteLine(""implementing system trait"");
	"));
		Assert.That(visitor.FileContent, Contains.Substring(@"	public bool Write(Generic generic)
	{
		Write(5) == true;
		var stringBuilder = ""printed successfully"";
		true;
	}"));
	}

	[Test]
	public void Import()
	{
		var interfaceType =
			new Type(package,
					new TypeLines(Computer, "has inputValue = 5", "has log", "Run", "\tlog.Write(inputValue)")).
				ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("namespace " + package.Name + ";"));
		Assert.That(visitor.FileContent, Contains.Substring("public class " + Computer));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic void Run()" + Environment.NewLine));
	}

	[Test]
	public void MemberInitializer()
	{
		var program =
			new Type(package,
				new TypeLines(Computer, "has number", "has file = \"test.txt\"", "Run",
					"\tfile.Write(number)",
					"\t6")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("public class " + Computer));
		Assert.That(visitor.FileContent, Contains.Substring("\tprivate int number"));
		Assert.That(visitor.FileContent,
			Contains.Substring(
				"\tprivate static FileStream file = new File(\"test.txt\", FileMode.OpenOrCreate);"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic void Run()" + Environment.NewLine));
	}

	[Test]
	public void LocalMemberNotFound() =>
		Assert.That(
			() => new CSharpTypeVisitor(
				new Type(package,
					new TypeLines(Computer, "has log", "Run", "\tconstant random = log.unknown")).ParseMembersAndMethods(parser)),
			Throws.InstanceOf<MethodExpressionParser.MemberOrMethodNotFound>());

	[Test]
	public void AccessLocalVariableAfterDeclaration() =>
		Assert.That(
			new CSharpTypeVisitor(
				new Type(package,
					new TypeLines(Computer, "has log", "has file", "Run", "\tconstant random = \"test\"",
						"\tlog.Write(random)")).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring("\tConsole.WriteLine(random);"));

	[TestCase(@"	constant file = File(""test.txt"")
	file.Write(number)", "\tvar file = new File(\"test.txt\", FileMode.OpenOrCreate);")]
	[TestCase(@"	File(""test"").Write(number)",
		"\tnew File(\"test\", FileMode.OpenOrCreate).Write(number);")]
	public void InitializeValueUsingConstructorInsideMethod(string code, string expected) =>
		Assert.That(new CSharpTypeVisitor(new Type(package, new TypeLines(Computer, (@"has number
Run
" + code).Split(Environment.NewLine))).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring(expected));

	[TestCase("ll + mm", "ll + mm")]
	[TestCase("ll - mm", "ll - mm")]
	[TestCase("ll * mm", "ll * mm")]
	public void ListsBinaryOperation(string code, string expected) =>
		Assert.That(new CSharpTypeVisitor(new Type(new TestPackage(), new TypeLines(Computer, @$"has log
Run
	constant ll = (1, 2) + (3, 4)
	constant mm = (5, 6)
	constant rr = {
		code
	}".Split(Environment.NewLine))).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring($"\tvar rr = {expected};"));

	[Test]
	public void GenerateListTypeProgram()
	{
		var program =
			new Type(new TestPackage(),
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
		Assert.That(new CSharpTypeVisitor(program).FileContent, Contains.Substring(
			@"namespace SourceGeneratorTests;

public class Program
{
	private System system;
	public int NestedMethod()
	{
		NestedMethod() == 5;
		if (5 == 5)
			if (5 is not 6)
				var aa = 5;
			else
				var bb = 5;
	}
}"));
	}
}