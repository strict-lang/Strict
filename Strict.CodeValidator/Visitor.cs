using Strict.Language;

namespace Strict.CodeValidator;

public abstract class Visitor
{
	protected Expression? Find<ExpressionType>(Expression searchIn, Func<ExpressionType, Expression?> findMatch)
	{
		var result = searchIn is ExpressionType matchedExpression
			? findMatch(matchedExpression)
			: null;
		if (result != null)
			return result;
		if (searchIn is Body body)
			foreach (var child in body.Expressions)
			{
				var foundChildMatch = Find(child, findMatch);
				if (foundChildMatch != null)
					return foundChildMatch;
			}
		return null;
	}

	public abstract Expression? Visit(Expression searchIn);
}