using System;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

[Ignore("TODO: Not yet done")]
public sealed class CSharpTypeVisitorTests : TestCSharpGenerator
{
	[Ignore("TODO: Not yet done")]
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
	public void GenerateInterface()
	{
		var interfaceType =
			new Type(package, new TypeLines(Computer, "Compute(number)")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("public interface " + Computer));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tvoid Compute(int number);" + Environment.NewLine));
	}

	private const string Computer = "Computer";

	[Test]
	public void GenerateTypeThatImplementsMultipleTraits()
	{
		var program = new Type(package,
				new TypeLines("Program", "implement Input", "implement Output", "has system", "Read",
					"\tsystem.WriteLine(\"Read\")", "Write", "\tsystem.WriteLine(\"Write\")")).
			ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent, Contains.Substring(@"	public void Read()
	{
		Console.WriteLine(""Read"");
	}"));
		Assert.That(visitor.FileContent, Contains.Substring(@"	public void Write()
	{
		Console.WriteLine(""Write"");
	}"));
	}

	[Ignore("TODO: Not yet done")]
	[Test]
	public void Import()
	{
		var interfaceType =
			new Type(package,
					new TypeLines(Computer, "has number", "has log", "Run", "\tlog.Write(number)")).
				ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("using Strict;"));
		Assert.That(visitor.FileContent, Contains.Substring("namespace " + package.Name + ";"));
		Assert.That(visitor.FileContent, Contains.Substring("public class " + Computer));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic void Run()" + Environment.NewLine));
	}

	[Ignore("TODO: Not yet done")]
	[Test]
	public void MemberInitializer()
	{
		var interfaceType =
			new Type(package,
				new TypeLines(Computer, "has number", "has file = \"test.txt\"", "Run",
					"\tfile.Write(number)")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("public class " + Computer));
		Assert.That(visitor.FileContent, Contains.Substring("\tprivate int number;"));
		Assert.That(visitor.FileContent,
			Contains.Substring(
				"\tprivate static FileStream file = new FileStream(\"test.txt\", FileMode.OpenOrCreate);"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic void Run()" + Environment.NewLine));
	}

	public void LocalMemberNotFound() =>
		Assert.That(
			() => new CSharpTypeVisitor(
				new Type(package,
					new TypeLines(Computer, "has log", "Run", "", "\tlet random = \"test\"",
						"\tlog.Write(random)")).ParseMembersAndMethods(parser)),
			Throws.InstanceOf<MethodExpressionParser.MemberOrMethodNotFound>()!);

	[Ignore("TODO: Not yet done")]
	[Test]
	public void AccessLocalVariableAfterDeclaration() =>
		Assert.That(
			new CSharpTypeVisitor(
				new Type(package,
					new TypeLines(Computer, "has log", "Run", "\tlet random = \"test\"",
						"\tlog.Write(random)")).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring("\tConsole.WriteLine(random);"));

	[Ignore("TODO: Not yet done")]
	[TestCase(@"	let file = File(""test.txt"")
	file.Write(number)", "\tvar file = new FileStream(\"test.txt\", FileMode.OpenOrCreate);")]
	[TestCase(@"	File(""test.txt"").Write(number)",
		"\tnew FileStream(\"test.txt\", FileMode.OpenOrCreate).Write(number);")]
	public void InitializeValueUsingConstructorInsideMethod(string code, string expected) =>
		Assert.That(new CSharpTypeVisitor(new Type(package, new TypeLines(Computer, (@"has number
Run
" + code).Split(Environment.NewLine))).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring(expected));

	[TestCase("l + m", "l + m")]
	[TestCase("l - m", "l - m")]
	[TestCase("l * m", "l * m")]
	public void ListsBinaryOperation(string code, string expected) =>
		Assert.That(new CSharpTypeVisitor(new Type(package, new TypeLines(Computer, @$"has log
Run
	let l = (1, 2) + (3, 4)
	let m = (5, 6)
	let r = {
		code
	}".Split(Environment.NewLine))).ParseMembersAndMethods(parser)).FileContent,
			Contains.Substring($"\tvar r = {expected};"));

	[Ignore("TODO: Not yet done")]
	[Test]
	public void GenerateListTypeProgram()
	{
		var program =
			new Type(package,
				new TypeLines("Program", "has numbers", "TestListsMethod returns Numbers",
					"\t(1, 2, 3) + 5", "\treturn numbers")).ParseMembersAndMethods(parser);
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent, Contains.Substring(@"	private List<int> numbers;"));
	}

	[Test]
	public void AssignmentWithMethodCall()
	{
		// @formatter:off
		var program = new Type(package,
			new TypeLines("Program",
				"implement App",
				"MethodToCall Text",
				"\t\"Hello World\"",
				"Run",
				"\tlet result = MethodToCall")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ToString(), Is.EqualTo("MethodToCall Text"));
		Assert.That(program.Methods[1].GetBodyAndParseIfNeeded().ToString(), Is.EqualTo("let result = MethodToCall"));
	}

	[Test]
	public void LocalMethodCallShouldHaveCorrectReturnType()
	{
		var program = new Type(package,
			new TypeLines("Program",
				"implement App",
				"LocalMethod Text",
				"\t\"Hello World\"",
				"Run",
				"\t\"Random Text\"")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ReturnType.Name, Is.EqualTo(Base.Text));
	}

	[Test]
	public void LetAssignmentWithConstructorCall()
	{
		var program = new Type(package,
			new TypeLines("Program", "implement App", "Run", "\tlet file = File(\"test.txt\")")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ToString(), Is.EqualTo("let file = File(\"test.txt\")"));
		Assert.That(program.Methods[0].ReturnType.Name, Is.EqualTo("File"));
	}

	[Test]
	public void AssignmentWithConstructorCall()
	{
		var program = new Type(package,
			new TypeLines("Program", "implement App", "has file = \"test.txt\"", "Run", "\tlet a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].Value?.ToString(), Is.EqualTo("File(\"test.txt\")"));
		Assert.That(program.Members[0].Value?.ReturnType.Name, Is.EqualTo("File"));
	}

	[Ignore("We don't allow non constructor expressions atm in has member")]
	[Test]
	public void AssignmentUsingGlobalMemberCall()
	{
		var program = new Type(package,
			new TypeLines("Program", "implement App", "has file = \"test.txt\"", "has fileDescription = file.Length > 1000 ? \"big file\" : \"small file\"", "Run", "\tlet a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[1].Name, Is.EqualTo("fileDescription"));
		Assert.That(program.Members[1].Value, Is.EqualTo("file.Length > 1000 ? \"big file\" : \"small file\""));
	}
}