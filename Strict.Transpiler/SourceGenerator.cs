using Type = Strict.Language.Type;

namespace Strict.Transpiler;

public interface SourceGenerator
{
	SourceFile Generate(Type app);
}