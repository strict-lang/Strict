using Type = Strict.Language.Type;

namespace Strict.Transpiler.Roslyn;

public class CSharpFile(Type type) : SourceFile
{
	private readonly CSharpTypeVisitor visitor = new(type);
	public override string ToString() => visitor.FileContent;
}