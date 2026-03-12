using Type = Strict.Language.Type;

namespace Strict.Transpiler.Roslyn;

public sealed class CSharpGenerator : SourceGenerator
{
	public SourceFile Generate(Type type) => new CSharpFile(type);
}