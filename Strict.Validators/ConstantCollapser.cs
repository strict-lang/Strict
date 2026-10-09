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
		var isComputedConstant = member.InitialValue is { IsConstant: true } and not Value and not
			MethodCall { Method.Name: Method.From, Arguments: [Value] };
		base.Visit(member, context);
		if (isComputedConstant && !member.IsConstant)
			throw new UseConstantHere(member.DefinedIn,
				member.DefinedIn.FindLineNumber(Type.HasWithSpaceAtEnd + member.Name));
	}

	/// <summary>
	/// A literal default is an optional constructor argument, a computed constant should be constant.
	/// </summary>
	public class UseConstantHere(Type type, int lineNumber) : ParsingFailed(type, lineNumber);

	protected override void Visit(Body body, object? context = null)
	{
		base.Visit(body, context);
		var rewritten = RemoveAllConstantDeclarations(body);
		if (rewritten != null)
			body.SetExpressions(rewritten);
	}

	private List<Expression>? RemoveAllConstantDeclarations(Body body)
	{
		List<Expression>? rewritten = null;
		for (var i = 0; i < body.Expressions.Count; i++)
			if (body.Expressions[i] is Declaration decl && decl.Value is not MethodCall &&
				!IsVariableStillUsed(body, decl.Name, i))
			{
				CollapsedCount++;
				if (rewritten == null)
				{
					rewritten = new List<Expression>(body.Expressions.Count - 1);
					for (var j = 0; j < body.Expressions.Count; j++)
						if (i != j)
							rewritten.Add(body.Expressions[j]);
				}
				else
				{
					rewritten.Remove(body.Expressions[i]);
				}
			}
		return rewritten;
	}

	private static bool IsVariableStillUsed(Body body, string variableName, int declarationIndex)
	{
		for (var i = 0; i < body.Expressions.Count; i++)
			if (i != declarationIndex && ContainsVariableCall(body.Expressions[i], variableName))
				return true; //ncrunch: no coverage
		return false;
	}

	private static bool ContainsVariableCall(Expression expression, string name) =>
		expression switch
		{
			//ncrunch: no coverage start
			VariableCall variableCall => variableCall.Variable.Name == name,
			MemberCall memberCall => memberCall.Instance != null &&
				ContainsVariableCall(memberCall.Instance, name),
			ListCall listCall => ContainsVariableCall(listCall.List, name) ||
				ContainsVariableCall(listCall.Index, name),
			MethodCall methodCall => (methodCall.Instance != null &&
					ContainsVariableCall(methodCall.Instance, name)) ||
				methodCall.Arguments.Any(argument => ContainsVariableCall(argument, name)),
			Declaration declaration => ContainsVariableCall(declaration.Value, name),
			MutableReassignment reassignment => ContainsVariableCall(reassignment.Target, name) ||
				ContainsVariableCall(reassignment.Value, name),
			For loop => ContainsVariableCall(loop.Iterator, name) || ContainsVariableCall(loop.Body, name),
			If branch => ContainsVariableCall(branch.Condition, name) ||
				ContainsVariableCall(branch.Then, name) || (branch.OptionalElse != null &&
					ContainsVariableCall(branch.OptionalElse, name)),
			Return returnExpression => ContainsVariableCall(returnExpression.Value, name),
			Body nested => nested.Expressions.Any(nestedExpression =>
				ContainsVariableCall(nestedExpression, name)),
			SelectorIf selector => ContainsVariableCall(selector.Selector, name) ||
				selector.Cases.Any(selectorCase => ContainsVariableCall(selectorCase.Pattern, name) ||
					ContainsVariableCall(selectorCase.Then, name)) || (selector.OptionalElse != null &&
					ContainsVariableCall(selector.OptionalElse, name)),
			List list => list.Values.Any(value => ContainsVariableCall(value, name)),
			//ncrunch: no coverage end
			_ => false
		};

	public int CollapsedCount { get; private set; }

	protected override Expression? Visit(Expression? expression, Body? body, object? context = null)
	{
		expression = base.Visit(expression, body, context);
		if (expression == null)
			return expression;
		if (expression is Binary binary)
		{
			var left = binary.Instance!;
			if (left is VariableCall { Variable: { IsMutable: false, InitialValue.IsConstant: true } } leftCall)
				left = leftCall.Variable.InitialValue;
			if (left is MemberCall { Member: { IsMutable: false, InitialValue.IsConstant: true } } leftMember)
				left = leftMember.Member.InitialValue;
			var right = binary.Arguments[0];
			if (right is VariableCall { Variable: { IsMutable: false, InitialValue.IsConstant: true } } rightCall)
				right = rightCall.Variable.InitialValue;
			if (right is MemberCall { Member: { IsMutable: false, InitialValue.IsConstant: true } } rightMember)
				right = rightMember.Member.InitialValue;
			var collapsedExpression = TryCollapseBinaryExpression(left, right, binary.Method);
			if (collapsedExpression != null)
				return collapsedExpression;
			if (!ReferenceEquals(left, binary.Instance!) || !ReferenceEquals(right, binary.Arguments[0]))
			{
				CollapsedCount++;
				var arguments = new[] { right };
				return new Binary(left, left.ReturnType.GetMethod(binary.Method.Name, arguments),
					arguments);
			}
		}
		if (!expression.IsConstant)
			return expression;
		return expression is To to && TryCollapseTo(to) is { } collapsed
			? collapsed
			: expression;
	}

	private Expression? TryCollapseTo(To to)
	{
		Expression? collapsed = to.Instance switch
		{
			Text textValue when to.ConversionType.IsNumber =>
				new Number(to.Method.Type, double.Parse(textValue.Data.Text)),
			Number numberValue when to.ConversionType.IsText =>
				new Text(to.Method.Type, numberValue.Data.ToExpressionCodeString()),
			Boolean boolValue when to.ConversionType.IsText =>
				new Text(to.Method.Type, boolValue.Data.Boolean ? "true" : "false"),
			_ => null
		};
		if (collapsed != null)
			CollapsedCount++;
		return collapsed;
	}

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
			if (leftText != null && right is Text rightText)
				return new Text(method, leftText.Data.Text + rightText.Data.Text);
			if (leftText != null && rightNumber != null)
				return new Text(method, leftText.Data.Text + rightNumber.Data.ToExpressionCodeString());
			if (leftText != null && right is Boolean rightBool)
				return new Text(method, leftText.Data.Text + rightBool.Data.Boolean);
		}
		else if (method.Name == BinaryOperator.Minus && leftNumber != null && rightNumber != null)
		{
			return new Number(method, leftNumber.Data.Number - rightNumber.Data.Number);
		}
		else if (method.Name == BinaryOperator.Multiply && leftNumber != null && rightNumber != null)
		{
			return new Number(method, leftNumber.Data.Number * rightNumber.Data.Number);
		}
		else if (method.Name == BinaryOperator.Divide && leftNumber != null && rightNumber != null)
		{
			return new Number(method, leftNumber.Data.Number / rightNumber.Data.Number);
		}
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