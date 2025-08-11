#if TODO //when visitors are fixed!
using Strict.Language;
using Strict.Expressions;

namespace Strict.Validators;

public sealed record ExpressionOptimizer(IEnumerable<Method> Methods) : Validator
{
	public void Validate()
	{
		foreach (var method in Methods)
			Validate(method);
	}

	private static void Validate(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded() as Body;
		if (body == null)
			return;
		var constants = new Dictionary<string, object?>();
		ValidateBody(body, constants);
	}

	private static void ValidateBody(Body body, Dictionary<string, object?> constants)
	{
		foreach (var expr in body.Expressions)
		{
			if (expr is ConstantDeclaration constant)
			{
				var value = TryEvaluate(constant.Value, constants);
				constants[constant.Name] = value;
				if (constant.Value is To toExpr)
				{
					if (value is null)
						throw new ImpossibleConstantCast();
				}
			}
			else if (expr is Body childBody)
			{
				ValidateBody(childBody, new Dictionary<string, object?>(constants));
			}
		}
	}

	private static object? TryEvaluate(Expression expr, Dictionary<string, object?> constants)
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
				return double.TryParse(s, out var num)
					? num
					: null;
			return leftVal;
		}
		return null;
	}

	public sealed class ImpossibleConstantCast : Exception;
}
#endif