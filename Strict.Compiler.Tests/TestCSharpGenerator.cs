using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Expressions;
using Strict.Expressions.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests;

public class TestCSharpGenerator : NoConsoleWriteLineAllowed
{
	[SetUp]
	public Task CreateGenerator()
	{
		parser = new MethodExpressionParser();
		package = new Package(nameof(SourceGeneratorTests));
		generator = new CSharpGenerator();
		return new Repositories(parser).LoadStrictPackage();
	}

	protected MethodExpressionParser parser = null!;
	protected Package package = null!;
	protected SourceGenerator generator = null!;

	protected Type CreateHelloWorldProgramType() =>
		new Type(package,
				new TypeLines("Program", "has App", "has logger", "Run",
					"\tlogger.Log(\"Hello World\")")).
			ParseMembersAndMethods(parser);
}