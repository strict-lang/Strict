using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.CodeValidator;

public sealed class MutableAssignmentVisitor(NamedType typeToSearchFor) : Visitor
{
	public override Expression? Visit(Expression searchIn) =>
		Find<MutableAssignment>(searchIn, expression =>
			expression.Name == typeToSearchFor.Name && expression.ReturnType == typeToSearchFor.Type
				? expression
				: null);
}