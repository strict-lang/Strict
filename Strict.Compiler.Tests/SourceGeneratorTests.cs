using System.Threading.Tasks;
using NUnit.Framework;
using Strict.Compiler.Roslyn;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Tests
{
	[Ignore("first finish LoadStrictBaseTypes")]
	public class SourceGeneratorTests
	{
		[SetUp]
		public async Task CreateGenerator()
		{
			await new Repositories().LoadFromUrl(Repositories.StrictUrl);
			package = new Package(nameof(SourceGeneratorTests));
			generator = new CSharpGenerator();
		}

		private Package package;
		private SourceGenerator generator;

		[Test]
		public void GenerateCSharpInterface()
		{
			var app = new Type(package, "DummyApp", "Run");
			var file = generator.Generate(app);
			Assert.That(file.ToString(), Is.EqualTo(@"public interface DummyApp
{
	void Run();
}"));
		}

		[Test]
		public void GenerateCSharpClass()
		{
			var program = new Type(package, "Program", @"implement App
has log
Run
	log.Write(""Hello World"")");
			var file = generator.Generate(program);
			Assert.That(file.ToString(), Is.EqualTo(@"public class Program
{
	public static void Main()
	{
		Console.WriteLine(""Hello World"");
	}
}"));
		}
	}
}