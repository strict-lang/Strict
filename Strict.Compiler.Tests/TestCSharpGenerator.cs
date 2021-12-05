using System.Threading.Tasks;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.Compiler.Tests;

public class TestCSharpGenerator
{
	[SetUp]
	public async Task CreateGenerator()
	{
		parser = new MethodExpressionParser();
		await new Repositories(parser).LoadFromUrl(Repositories.StrictUrl);
		package = new Package(nameof(SourceGeneratorTests));
		generator = new CSharpGenerator();
	}

	protected MethodExpressionParser parser = null!;
	protected Package package = null!;
	protected SourceGenerator generator = null!;

	protected Type CreateHelloWorldProgramType() =>
		new Type(package, "Program", parser).Parse(@"implement App
has log
Run
	log.Write(""Hello World"")");
}