using System.Globalization;
using Boolean = Strict.Expressions.Boolean;

namespace Strict.Validators;

/// <summary>
/// Reduces constant expressions, e.g. "5" to Number can just be 5. Or any binary expression like
/// 2 + 3 can be reduced to 5 as long as both sides are constant. This is done recursively and all
/// usages will be replaced by the constant and folded further until no more constants exist.
/// </summary>
public sealed class ConstantCollapser : Visitor
{
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

	protected override Expression? Visit(Expression? expression, Body body, object? context = null)
	{
		var processedExpression = base.Visit(expression, body, context);
		if (expression is Binary binary)
		{
			var left = binary.Instance!;
			if (left is VariableCall { Variable.InitialValue.IsConstant: true } leftCall)
				left = leftCall.Variable.InitialValue;
			var right = binary.Arguments[0];
			if (right is VariableCall { Variable.InitialValue.IsConstant: true } rightCall)
				right = rightCall.Variable.InitialValue;
			var collapsedExpression = TryCollapseBinaryExpression(left, right, binary.Method);
			if (collapsedExpression != null)
				return collapsedExpression;
			if (!ReferenceEquals(left, binary.Instance!) || !ReferenceEquals(right, binary.Arguments[0]))
			{
				var arguments = new[] { right };
				return new Binary(left, left.ReturnType.GetMethod(binary.Method.Name, arguments), arguments);
			}
		}
		return processedExpression;
	}

	/// <summary>
	/// Would be nice if all of these are evaluated via actual strict code!
	/// </summary>
	private static Expression? TryCollapseBinaryExpression(Expression left, Expression right,
		Context method)
	{
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
			if (leftText != null && right is Boolean rightBoolean)
				return new Text(method, (string)leftText.Data + rightBoolean.Data);
			if (left is Boolean leftBoolean && rightText != null)
				return new Text(method, leftBoolean.Data + (string)rightText.Data);
		}
		else if (method.Name == BinaryOperator.Minus && leftNumber != null && rightNumber != null)
			return new Number(method, (double)leftNumber.Data - (double)rightNumber.Data);
		else if (method.Name == BinaryOperator.Multiply && leftNumber != null && rightNumber != null)
			return new Number(method, (double)leftNumber.Data * (double)rightNumber.Data);
		else if (method.Name == BinaryOperator.Divide && leftNumber != null && rightNumber != null)
			return new Number(method, (double)leftNumber.Data / (double)rightNumber.Data);
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
			throw new NotSupportedException("TODO");
		}
		return expression;
	}

	/*TODO
	public override void Visit(Expression? expression, object? context = null)
	{
		if (context is not Dictionary<string, object?> constants)
			return;
		if (expression is ConstantDeclaration constant)
		{
			var value = TryEvaluate(constant.Value, constants);
			constants[constant.Name] = value;
			if (constant.Value is To toExpr)
			{
				if (value is null)
					throw new ImpossibleConstantCast();
			}
		}
	}

	/*this seems to be nonsense gibberish, has to be checked		*
	private object? TryEvaluate(Expression expr, Dictionary<string, object?> constants)
	{
		// Handle Number, Text, Boolean
		if (expr is Value v)
			return v.Data;
		// Handle variable propagation
		if (expr is ParameterCall param && constants.TryGetValue(param.Parameter.Name, out var val))
			return val;
		// Handle simple binary operations
		if (expr is Binary binary)
		{
			var left = TryEvaluate(binary.Instance!, constants);
			var right = TryEvaluate(binary.Arguments[0], constants);
			if (left is double l && right is double r)
				switch (binary.Method.Name)
				{
				case "+":
					return l + r;
				case "-":
					return l - r;
				case "*":
					return l * r;
				case "/":
					return l / r;
				}
		}
		// Handle To (cast)
		if (expr is To to)
		{
			var leftVal = TryEvaluate(to.Instance!, constants);
			if (to.ConversionType.Name == "Number" && leftVal is string s)
				return double.Parse(s);
			return leftVal;
		}
		return null;
	}

	public sealed class ImpossibleConstantCast : Exception;
	*/

	/*gibberish again
	// Returns true if compile-time constant; value is the evaluated Value.Data (double/string/bool)
	public static bool IsConstant(Expression expr, ConstantEnv env, out object? value)
	{
		value = null;

		// 1) Literal constants
		if (expr is Value v) { value = v.Data; return true; }

		// 2) Variable/Member/Parameter references
		if (expr is VariableCall vc)
			return env.Locals.TryGetValue(vc.Variable.Name, out value);
		if (expr is MemberCall mc)
			return env.Members.TryGetValue(mc.Member.Name, out value);
		if (expr is ParameterCall) // unknown at definition site
			return false;

		// 3) Casts
		if (expr is To to)
		{
			if (!TryGetConstant(to.Instance!, env, out var inner))
				return false;
			if (to.ConversionType.Name == Base.Number)
			{
				if (inner is string s && double.TryParse(s, out var n)) { value = n; return true; }
				if (inner is double d) { value = d; return true; }
				return false; // impossible cast → handled by validator/optimizer elsewhere
			}
			if (to.ConversionType.Name == Base.Text)
			{
				value = inner switch
				{
					string s => s,
					double d => d.ToString(CultureInfo.InvariantCulture),
					bool b => b.ToString(),
					_ => null
				};
				return value is not null;
			}
			return false;
		}

		// 4) Binary math on Numbers (extend as needed)
		if (expr is Binary bin &&
			TryGetConstant(bin.Instance!, env, out var l) &&
			TryGetConstant(bin.Arguments[0], env, out var r) &&
			l is double ld && r is double rd)
		{
			value = bin.Method.Name switch
			{
				"+" => ld + rd,
				"-" => ld - rd,
				"*" => ld * rd,
				"/" => rd == 0 ? null : ld / rd,
				_ => null
			};
			return value is not null;
		}

		// 5) Control flow (optional): If condition constant -> pick branch and recurse
		if (expr is If iff && TryGetConstant(iff.Condition, env, out var cond) && cond is bool b)
			return TryGetConstant(b ? iff.Then : (iff.OptionalElse ?? iff.Then), env, out value);

		// 6) Collections (optional): List/Dictionary constant if all elements constant

		return false;
	}
	*/

	/*this seems to be nonsense gibberish, has to be checked

		protected override Expression Visit(Expression? expression, object? context = null)
		{
			if (expression is ConstantDeclaration constant)
			{
				constant.Value = Collapse(constant.Value);
				constants[constant.Name] = value;
				if (constant.Value is To toExpr)
				{
					if (value is null)
						throw new ImpossibleConstantCast();
				}
			}
		}

	private object? Collapse(Expression expression)
	{
		// Handle Number, Text, Boolean
		if (expr is Value v)
			return v.Data;
		// Handle variable propagation
		if (expr is ParameterCall param && constants.TryGetValue(param.Parameter.Name, out var val))
			return val;
		// Handle simple binary operations
		if (expr is Binary binary)
		{
			var left = TryEvaluate(binary.Instance!, constants);
			var right = TryEvaluate(binary.Arguments[0], constants);
			if (left is double l && right is double r)
				switch (binary.Method.Name)
				{
				case "+":
					return l + r;
				case "-":
					return l - r;
				case "*":
					return l * r;
				case "/":
					return l / r;
				}
		}
		// Handle To (cast)
		if (expr is To to)
		{
			var leftVal = TryEvaluate(to.Instance!, constants);
			if (to.ConversionType.Name == "Number" && leftVal is string s)
				return double.Parse(s);
			return leftVal;
		}
		return null;
	}

	public sealed class ImpossibleConstantCast : Exception;
	*/
}