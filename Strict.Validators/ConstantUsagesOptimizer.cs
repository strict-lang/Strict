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
		context ??= new Dictionary<string, object?>();
		base.Visit(body, context);
	}

	protected override Expression VisitExpression(Expression expression, object? context)
	{
		if (context is not Dictionary<string, object?> constants)
			return expression;
		//TODO: finish this up!
		//if (expression.IsConstant)
		//	constants.Add();
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