using Strict.Expressions;
using Strict.Language;

namespace Strict.Validators;

public sealed class MutableAssignmentVisitor(NamedType typeToSearchFor) : Visitor
{
	public override Expression? Visit(Expression searchIn) =>
		Find<MutableReassignment>(searchIn, expression =>
			expression.Name == typeToSearchFor.Name && expression.ReturnType == typeToSearchFor.Type
				? expression
				: null);
}