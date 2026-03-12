using NUnit.Framework;
using Strict.Transpiler.Roslyn;
using Strict.Language;
using Strict.Expressions;
using Strict.Expressions.Tests;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Transpiler.Tests;

public class TestCSharpGenerator : NoConsoleWriteLineAllowed
{
	[SetUp]
	public void CreateGenerator()
	{
		parser = new MethodExpressionParser();
		package = new Package(nameof(SourceGeneratorTests));
		_ = TestPackage.Instance;
		new Type(package, new TypeLines(Type.App, "Run")).ParseMembersAndMethods(parser);
		new Type(package,
				new TypeLines(Type.System, "has textWriter", "Write(text)", "\ttextWriter.Write(text)")).
			ParseMembersAndMethods(parser);
		new Type(package, new TypeLines("Input", "Read Text")).ParseMembersAndMethods(parser);
		new Type(package, new TypeLines("Output", "Write(generic) Boolean")).
			ParseMembersAndMethods(parser);
		generator = new CSharpGenerator();
	}

	[TearDown]
	public void DisposePackage() => package.Dispose();

	protected MethodExpressionParser parser = null!;
	protected Package package = null!;
	protected SourceGenerator generator = null!;

	protected Type CreateHelloWorldProgramType() =>
		new Type(package,
			new TypeLines("Program", "has App", "has logger", "Run",
				"\tlogger.Log(\"Hello World\")")).ParseMembersAndMethods(parser);
}