using Type = Strict.Language.Type;

namespace Strict.Compiler;

public interface SourceGenerator
{
	SourceFile Generate(Type app);
}