using Strict.Language.Expressions;

namespace Strict.Compiler.Roslyn;

public interface ExpressionVisitor
{
	string Visit(int tabIndentation = 2);

	string Visit(MemberCall call);
	//TODO: implement to satisfy File tests in SourceGeneratorTests
}