using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public class CSharpFile : SourceFile
{
	public CSharpFile(Type type) => visitor = new CSharpTypeVisitor(type);
	private readonly CSharpTypeVisitor visitor;
	public override string ToString() => visitor.FileContent;
}