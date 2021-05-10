using Strict.Language;

namespace Strict.Compiler.Roslyn
{
    public class CSharpGenerator : SourceGenerator
    {
        public SourceFile Generate(Type type) => new CSharpFile(type);
    }
}