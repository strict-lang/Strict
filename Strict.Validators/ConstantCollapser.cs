using System.Globalization;
using Boolean = Strict.Expressions.Boolean;
using Type = Strict.Language.Type;

namespace Strict.Validators;

/// <summary>
/// Reduces constant expressions, e.g. "5" to Number can just be 5. Or any binary expression like
/// 2 + 3 can be reduced to 5 as long as both sides are constant. This is done recursively, and
/// all usages will be replaced by the constant and folded further until no more constants exist.
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
			if (body.Expressions[i] is Declaration)
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
		if (expression == null)
			return expression;
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
		if (!expression.IsConstant)
			return expression;
		if (expression is To to)
		{
			var value = to.Instance as Value;
			if (to.ConversionType.IsNumber && value is Text textValue)
				return new Number(to.Method.Type, double.Parse(textValue.Data.Text));
			if (to.ConversionType.IsText && value is Number numberValue)
				return new Text(to.Method.Type, numberValue.Data.Number.ToString(CultureInfo.InvariantCulture));
			throw new UnsupportedToExpression(to.ToStringWithType()); //ncrunch: no coverage
		}
		return expression;
	}

	public class UnsupportedToExpression(string toStringWithType) : Exception(toStringWithType); //ncrunch: no coverage

	private static Expression? TryCollapseBinaryExpression(Expression left, Expression right,
		Context method)
	{
		if (left is Binary leftBinary)
			left = TryCollapseBinaryExpression(leftBinary.Instance!, leftBinary.Arguments[0],
				leftBinary.Method) ?? left;
		if (right is Binary rightBinary)
			right = TryCollapseBinaryExpression(rightBinary.Instance!, rightBinary.Arguments[0],
				rightBinary.Method) ?? right;
		var leftNumber = left as Number;
		var rightNumber = right as Number;
		if (method.Name == BinaryOperator.Plus)
		{
			if (leftNumber != null && rightNumber != null)
				return new Number(method, leftNumber.Data.Number + rightNumber.Data.Number);
			var leftText = left as Text;
			var rightText = right as Text;
			if (leftText != null && rightText != null)
				return new Text(method, leftText.Data.Text + rightText.Data.Text);
			if (leftText != null && rightNumber != null)
				return new Text(method, leftText.Data.Text + rightNumber.Data.ToExpressionCodeString());
			if (leftText != null && right is Boolean rightBool)
				return new Text(method, leftText.Data.Text + rightBool.Data.Boolean);
		}
		else if (method.Name == BinaryOperator.Minus && leftNumber != null && rightNumber != null)
			return new Number(method, leftNumber.Data.Number - rightNumber.Data.Number);
		else if (method.Name == BinaryOperator.Multiply && leftNumber != null && rightNumber != null)
			return new Number(method, leftNumber.Data.Number * rightNumber.Data.Number);
		else if (method.Name == BinaryOperator.Divide && leftNumber != null && rightNumber != null)
			return new Number(method, leftNumber.Data.Number / rightNumber.Data.Number);
		if (left is Boolean leftBoolean && right is Boolean rightBoolean)
		{
			if (method.Name == BinaryOperator.And)
				return new Boolean(method, leftBoolean.Data.Boolean && rightBoolean.Data.Boolean);
			if (method.Name == BinaryOperator.Or)
				return new Boolean(method, leftBoolean.Data.Boolean || rightBoolean.Data.Boolean);
		}
		return null;
	}
}