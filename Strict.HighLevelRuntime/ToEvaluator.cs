using Strict.Expressions;
using System.Globalization;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ToEvaluator(Interpreter interpreter)
{
	public ValueInstance Evaluate(To to, ExecutionContext ctx)
	{
		interpreter.Statistics.ToConversionCount++;
		var left = interpreter.RunExpression(to.Instance!, ctx);
		if (to.Instance!.ReturnType.IsText && to.ConversionType.IsNumber && left.IsText)
			return new ValueInstance(to.ConversionType,
				double.Parse(left.Text, CultureInfo.InvariantCulture));
		if (to.ConversionType.IsText)
			return left.TryGetValueTypeInstance() is { } typeInstance
				? new ValueInstance(BuildMembersText(typeInstance))
				: new ValueInstance(left.ToExpressionCodeString());
		return to.Method.IsTrait
			? throw new ToMethodNotImplemented(left, to.ConversionType)
			: interpreter.methodCallEvaluator.Evaluate(to, ctx);
	}

	private static string BuildMembersText(ValueTypeInstance typeInstance)
	{
		var values = typeInstance.Values;
		if (values.Length == 0)
			return typeInstance.ReturnType.Name;
		var parts = new string[values.Length];
		for (var index = 0; index < values.Length; index++)
			parts[index] = values[index].ToExpressionCodeString();
		return "(" + string.Join(", ", parts) + ")";
	}

	//ncrunch: no coverage start
	public sealed class ToMethodNotImplemented(ValueInstance left, Type toConversionType)
		: ExecutionFailed(toConversionType, "Conversion from " + left + " to " +
			toConversionType.Name + " not supported");
}