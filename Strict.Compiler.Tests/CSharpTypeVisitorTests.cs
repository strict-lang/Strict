using System;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

/*TODO: fix me
[Ignore("TODO: flaky Strict.Language.Type+MustImplementAllTraitMethods : Missing methods: Strict.Base.Text.digits, Strict.Base.Text.+\r\n   at Strict.Base.Error Implements ")]
public sealed class CSharpTypeVisitorTests : TestCSharpGenerator
{
	[Test]
	[Ignore("TODO: flaky Strict.Language.Type+MustImplementAllTraitMethods : Missing methods: Strict.Base.Text.digits, Strict.Base.Text.+\r\n   at Strict.Base.Error Implements ")]
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
	[Ignore("TODO: flaky Strict.Language.Type+MustImplementAllTraitMethods : Missing methods: Strict.Base.Text.digits, Strict.Base.Text.+\r\n   at Strict.Base.Error Implements ")]
	public void GenerateInterface()
	{
		var interfaceType = new Type(package, Computer, parser).Parse(@"Compute(number)");
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("public interface " + Computer));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tvoid Compute(int number);" + Environment.NewLine));
	}

	private const string Computer = "Computer";

	[Test]
	[Ignore("TODO: flaky Strict.Language.Type+MustImplementAllTraitMethods : Missing methods: Strict.Base.Text.digits, Strict.Base.Text.+\r\n   at Strict.Base.Error Implements ")]
	public void GenerateTypeThatImplementsMultipleTraits()
	{
		var program = new Type(package, "Program", parser).Parse(@"implement Input
implement Output
has system
Read
	system.WriteLine(""Read"")
Write
	system.WriteLine(""Write"")");
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent,
			Contains.Substring(@"	public void Read()
	{
		Console.WriteLine(""Read"");
	}"));
		Assert.That(visitor.FileContent,
			Contains.Substring(@"	public void Write()
	{
		Console.WriteLine(""Write"");
	}"));
	}

	[Test]
	[Ignore("TODO: flaky Strict.Language.Type+MustImplementAllTraitMethods : Missing methods: Strict.Base.Text.digits, Strict.Base.Text.+\r\n   at Strict.Base.Error Implements ")]
	public void Import()
	{
		var interfaceType = new Type(package, Computer, parser).Parse(@"import Strict
has number
has log
Run
	log.Write(number)");
		var visitor = new CSharpTypeVisitor(interfaceType);
		Assert.That(visitor.Name, Is.EqualTo(Computer));
		Assert.That(visitor.FileContent, Contains.Substring("using Strict;"));
		Assert.That(visitor.FileContent, Contains.Substring("namespace " + package.Name + ";"));
		Assert.That(visitor.FileContent, Contains.Substring("public class " + Computer));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic void Run()" + Environment.NewLine));
	}

	[Test]
	public void MemberInitializer()
	{
		var interfaceType = new Type(package, Computer, parser).Parse(@"import Strict
has number
has file = ""test.txt""
Run
	file.Write(number)");
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

	[TestCase(@"
	let random = ""test""
	log.Write(randomm)")]
	[TestCase(@"
	log.Write(random)
	let random = ""test""")]
	public void LocalMemberNotFound(string code) =>
		Assert.That(() => new CSharpTypeVisitor(new Type(package, Computer, parser).Parse(@"import Strict
has log
Run" + code)),
			Throws.InstanceOf<MethodExpressionParser.MemberOrMethodNotFound>()!);

	//TODO: flaky test, sometimes fails with MissingMethodImplementation (trait not loaded yet)
	[Test]
	[Ignore("TODO: fix!")]
	public void AccessLocalVariableAfterDeclaration() =>
		Assert.That(
			new CSharpTypeVisitor(
				new Type(package, Computer, parser).Parse(@"import Strict
has log
Run
	let random = ""test""
	log.Write(random)")).FileContent,
			Contains.Substring("\tConsole.WriteLine(random);"));

	[TestCase(@"	let file = File(""test.txt"")
	file.Write(number)",
		"\tvar file = new FileStream(\"test.txt\", FileMode.OpenOrCreate);")]
	[TestCase(@"	File(""test.txt"").Write(number)",
		"\tnew FileStream(\"test.txt\", FileMode.OpenOrCreate).Write(number);")]
	[Ignore("TODO: flaky Strict.Language.Type+MustImplementAllTraitMethods : Missing methods: Strict.Base.Text.digits, Strict.Base.Text.+\r\n   at Strict.Base.Error Implements ")]
	public void InitializeValueUsingConstructorInsideMethod(string code, string expected) =>
		Assert.That(
			new CSharpTypeVisitor(new Type(package, Computer, parser).Parse(@"import Strict
has number
Run
" + code)).FileContent,
			Contains.Substring(expected));

	[TestCase("l + m", "l + m")]
	[TestCase("l - m", "l - m")]
	[TestCase("l * m", "l * m")]
	public void ListsBinaryOperation(string code, string expected) =>
		Assert.That(new CSharpTypeVisitor(new Type(package, Computer, parser).Parse(@$"import Strict
has log
Run
	let l = (1, 2) + (3, 4)
	let m = (5, 6)
	let r = {
		code
	}")).FileContent, Contains.Substring($"\tvar r = {expected};"));

	[Ignore("Have to do it next after constructors and generics")]
	[Test]
	public void GenerateListTypeProgram()
	{
		var program = new Type(package, "Program", parser).Parse(@"has numbers
TestListsMethod returns Numbers
	(1, 2, 3) + 5
	return numbers");
		var visitor = new CSharpTypeVisitor(program);
		AssertProgramClass(visitor);
		Assert.That(visitor.FileContent,
			Contains.Substring(@"	private List<int> numbers;"));
	}
}
*/