using Strict.Language;

namespace Strict.Compiler.Roslyn;

public class CSharpFile : SourceFile
{
	public CSharpFile(Type type)
	{
		this.type = type;
		visitor = new CSharpTypeVisitor(type);
	}

	private readonly Type type;
	private readonly CSharpTypeVisitor visitor;

	public override string ToString() => visitor.FileContent;
	/*don't need this hack anymore
		type.Name switch
		{
			"GenerateFileReadProgram" => @"public class Program
	{
		public static void Main()
		{
			Console.WriteLine(""Black friday is coming!"");
		}
	}",
			"DummyApp" => "public interface DummyApp\r\n{\r\n\tvoid Run();\r\n}",
			_ => @"public class Program
{
	public static void Main()
	{
		Console.WriteLine(""Hello World"");
	}
}"
		};*/
}