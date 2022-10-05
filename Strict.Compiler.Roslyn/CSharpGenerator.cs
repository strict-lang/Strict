using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public sealed class CSharpGenerator : SourceGenerator
{
	public SourceFile Generate(Type type) => new CSharpFile(type);
}