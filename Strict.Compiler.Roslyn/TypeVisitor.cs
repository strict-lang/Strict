using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public interface TypeVisitor
{
	void VisitImplement(Type type);
	void VisitMember(Member member);
	void VisitMethod(Method method);
	void ParsingDone();
}