using System;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

public sealed class CSharpTypeVisitorTests : TestGenerator
{
	[Test]
	public void GenerateHelloWorldApp()
	{
		var program = CreateHelloWorldProgramType();
		var visitor = new CSharpTypeVisitor(program);
		Assert.That(visitor.Name, Is.EqualTo("Program"));
		Assert.That(visitor.FileContent, Contains.Substring("public class Program"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\tpublic static void Main()" + Environment.NewLine + "\t{"));
		Assert.That(visitor.FileContent,
			Contains.Substring("\t\tConsole.WriteLine(\"Hello World\");"));
		Assert.That(visitor.FileContent.EndsWith("}", StringComparison.InvariantCulture), Is.True,
			visitor.FileContent);
	}

	[Test]
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

	//TODO: write more tests with many implements

	//TODO: test imports

	//TODO: test members
	//Assert.That(visitor.FileContent, Contains.Substring("\tprivate Log log;"));
}