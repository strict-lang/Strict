using Strict.Language;

namespace Strict.Compiler
{
	public interface SourceGenerator
	{
		SourceFile Generate(Type app);
	}
}