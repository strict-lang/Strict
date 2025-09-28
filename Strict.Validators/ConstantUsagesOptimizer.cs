using Strict.Expressions;

namespace Strict.Validators;

/// <summary>
/// Reduces constant expressions, e.g. "5" to Number can just be 5. Or any binary expression like
/// 2 + 3 can be reduced to 5 as long as both sides are constant. This is done recursively. After
/// this <see cref="ConstantCollapser"/> will take this a step further and collapse all usages.
/// </summary>
public sealed class ConstantUsagesOptimizer : Visitor
{
	protected override void Visit(Body body, object? context = null)
	{
		//context ??= new Dictionary<string, Expression>();
		base.Visit(body, context);
		// Remove all constant declarations that are never used
		List<Expression>? rewritten = null;
		for (var i = 0; i < body.Expressions.Count; i++)
			if (body.Expressions[i] is ConstantDeclaration constant)
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
		if (rewritten != null)
			body.SetExpressions(rewritten);
	}

	protected override Expression? Visit(Expression? expression, Body body, object? context = null)
	{
		/*
	protected override Expression VisitExpression(Expression expression, object? context)
	{
		if (context is not Dictionary<string, Expression> constants)
			return expression;
		Console.WriteLine(expression.GetType()+": "+ expression.ToString());
		if (expression.IsConstant)
		{
			constants.Add(expression.ToString(), expression);
		}
	*/
		if (expression is Binary binary)
		{
			var left = binary.Instance!;
			if (left is VariableCall { Variable.InitialValue.IsConstant: true } leftCall)
				left = leftCall.Variable.InitialValue;
			var right = binary.Arguments[0];
			if (right is VariableCall { Variable.InitialValue.IsConstant: true } rightCall)
				right = rightCall.Variable.InitialValue;
			//Console.WriteLine("left="+left+", right="+right+", binary="+binary);
			// ReSharper disable PossibleUnintendedReferenceComparison
			if (left != binary.Instance! || right != binary.Arguments[0])
			{
				var arguments = new[] { right };
				return binary.Method.Name switch
				{
					"+" => new Number(binary.Method,
						double.Parse(left.ToString()) + double.Parse(right.ToString())),
					"-" => new Number(binary.Method,
						double.Parse(left.ToString()) - double.Parse(right.ToString())),
					"*" => new Number(binary.Method,
						double.Parse(left.ToString()) * double.Parse(right.ToString())),
					"/" => new Number(binary.Method,
						double.Parse(left.ToString()) / double.Parse(right.ToString())),
					_ => new Binary(left, left.ReturnType.GetMethod(binary.Method.Name, arguments),
						arguments)
				};
			}
		}
		return base.Visit(expression, body, context);
	}

	protected override Expression VisitExpression(Expression expression, object? context)
	{
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
}