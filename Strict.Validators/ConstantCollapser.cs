using System.Globalization;
using Boolean = Strict.Expressions.Boolean;
using Type = Strict.Language.Type;

namespace Strict.Validators;

/// <summary>
/// Reduces constant expressions, e.g. "5" to Number can just be 5. Or any binary expression like
/// 2 + 3 can be reduced to 5 as long as both sides are constant. This is done recursively and all
/// usages will be replaced by the constant and folded further until no more constants exist.
/// </summary>
public sealed class ConstantCollapser : Visitor
{
	protected override void Visit(Member member, object? context = null)
	{
		base.Visit(member, context);
		if (member.InitialValue is { IsConstant: true } && !member.IsConstant)
			throw new UseConstantHere(member.Type,
				member.Type.FindLineNumber(Type.HasWithSpaceAtEnd + member.Name));
	}

	public class UseConstantHere(Type type, int lineNumber) : ParsingFailed(type, lineNumber);

	protected override void Visit(Body body, object? context = null)
	{
		base.Visit(body, context);
		var rewritten = RemoveAllConstantDeclarations(body);
		if (rewritten != null)
			body.SetExpressions(rewritten);
	}

	private static List<Expression>? RemoveAllConstantDeclarations(Body body)
	{
		List<Expression>? rewritten = null;
		for (var i = 0; i < body.Expressions.Count; i++)
			if (body.Expressions[i] is ConstantDeclaration)
			{
				if (rewritten == null)
				{
					rewritten = new List<Expression>(body.Expressions.Count - 1);
					for (var j = 0; j < body.Expressions.Count; j++)
						if (i != j)
							rewritten.Add(body.Expressions[j]);
				}
				else
					rewritten.Remove(body.Expressions[i]);
			}
		return rewritten;
	}

	protected override Expression? Visit(Expression? expression, Body? body, object? context = null)
	{
		expression = base.Visit(expression, body, context);
		if (expression is Binary binary)
		{
			var left = binary.Instance!;
			if (left is VariableCall { Variable.InitialValue.IsConstant: true } leftCall)
				left = leftCall.Variable.InitialValue;
			if (left is MemberCall { Member.InitialValue.IsConstant: true } leftMember)
				left = leftMember.Member.InitialValue;
			var right = binary.Arguments[0];
			if (right is VariableCall { Variable.InitialValue.IsConstant: true } rightCall)
				right = rightCall.Variable.InitialValue;
			if (right is MemberCall { Member.InitialValue.IsConstant: true } rightMember)
				right = rightMember.Member.InitialValue;
			var collapsedExpression = TryCollapseBinaryExpression(left, right, binary.Method);
			if (collapsedExpression != null)
				return collapsedExpression;
			if (!ReferenceEquals(left, binary.Instance!) || !ReferenceEquals(right, binary.Arguments[0]))
			{
				var arguments = new[] { right };
				return new Binary(left, left.ReturnType.GetMethod(binary.Method.Name, arguments), arguments);
			}
		}
		return expression;
	}

	/// <summary>
	/// Would be nice if all of these are evaluated via actual strict code!
	/// </summary>
	private static Expression? TryCollapseBinaryExpression(Expression left, Expression right,
		Context method)
	{
		if (left is Binary leftBinary)
			left = TryCollapseBinaryExpression(leftBinary.Instance!, leftBinary.Arguments[0], leftBinary.Method) ?? left;
		if (right is Binary rightBinary)
			right = TryCollapseBinaryExpression(rightBinary.Instance!, rightBinary.Arguments[0], rightBinary.Method) ?? right;
		var leftNumber = left as Number;
		var rightNumber = right as Number;
		if (method.Name == BinaryOperator.Plus)
		{
			if (leftNumber != null && rightNumber != null)
				return new Number(method, (double)leftNumber.Data + (double)rightNumber.Data);
			var leftText = left as Text;
			var rightText = right as Text;
			if (leftText != null && rightText != null)
				return new Text(method, (string)leftText.Data + (string)rightText.Data);
			if (leftText != null && rightNumber != null)
				return new Text(method, (string)leftText.Data + rightNumber.Data);
			if (leftNumber != null && rightText != null)
				return new Text(method, (double)leftNumber.Data + (string)rightText.Data);
			if (leftText != null && right is Boolean rightBool)
				return new Text(method, (string)leftText.Data + rightBool.Data);
			if (left is Boolean leftBool && rightText != null)
				return new Text(method, leftBool.Data + (string)rightText.Data);
		}
		else if (method.Name == BinaryOperator.Minus && leftNumber != null && rightNumber != null)
			return new Number(method, (double)leftNumber.Data - (double)rightNumber.Data);
		else if (method.Name == BinaryOperator.Multiply && leftNumber != null && rightNumber != null)
			return new Number(method, (double)leftNumber.Data * (double)rightNumber.Data);
		else if (method.Name == BinaryOperator.Divide && leftNumber != null && rightNumber != null)
			return new Number(method, (double)leftNumber.Data / (double)rightNumber.Data);
		if (left is Boolean leftBoolean && right is Boolean rightBoolean)
		{
			if (method.Name == BinaryOperator.And)
				return new Boolean(method, (bool)leftBoolean.Data && (bool)rightBoolean.Data);
			if (method.Name == BinaryOperator.Or)
				return new Boolean(method, (bool)leftBoolean.Data || (bool)rightBoolean.Data);
		}
		return null;
	}

	protected override Expression VisitExpression(Expression expression, object? context)
	{
		if (!expression.IsConstant)
			return expression;
		if (expression is To to)
		{
			var value = to.Instance as Value;
			if (to.ConversionType.Name == Base.Number && value?.Data is string text)
				return new Number(to.Method.Type, double.Parse(text));
			if (to.ConversionType.Name == Base.Text && value?.Data is double number)
				return new Text(to.Method.Type, number.ToString(CultureInfo.InvariantCulture));
			throw new UnsupportedToExpression(to.ToStringWithType());
		}
		return expression;
	}

	public class UnsupportedToExpression(string toStringWithType) : Exception(toStringWithType);
}