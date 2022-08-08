using System.Threading.Tasks;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.Compiler.Tests;

public class TestCSharpGenerator
{
	[SetUp]
	public Task CreateGenerator()
	{
		parser = new MethodExpressionParser();
		package = new Package(nameof(SourceGeneratorTests));
		generator = new CSharpGenerator();
		return new Repositories(parser).LoadFromUrl(Repositories.StrictUrl);
	}

	protected MethodExpressionParser parser = null!;
	protected Package package = null!;
	protected SourceGenerator generator = null!;

	protected Type CreateHelloWorldProgramType() =>
		new Type(package,
				new TypeLines("Program", "implement App", "has log", "Run",
					"\tlog.Write(\"Hello World\")")).
			ParseMembersAndMethods(parser);
}