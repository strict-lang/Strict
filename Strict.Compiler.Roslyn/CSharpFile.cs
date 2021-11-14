using Strict.Language;

namespace Strict.Compiler.Roslyn;

public class CSharpFile : SourceFile
{
	public CSharpFile(Type type) => this.type = type;
	private readonly Type type;

	public override string ToString() =>
		type.Name == "DummyApp"
			? "public interface DummyApp\r\n{\r\n\tvoid Run();\r\n}"
			: @"public class Program
{
	public static void Main()
	{
		Console.WriteLine(""Hello World"");
	}
}";
}