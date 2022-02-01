using Strict.Language.Expressions;

namespace Strict.Compiler.Roslyn;

public interface ExpressionVisitor
{
	string Visit(int tabIndentation = 2);
	string Visit(MemberCall member);
	string Visit(MethodCall call);
	string Visit(Text text);

	//TODO: support all of Strict.Language.Expressions
	//TODO: implement to satisfy File tests in SourceGeneratorTests
}