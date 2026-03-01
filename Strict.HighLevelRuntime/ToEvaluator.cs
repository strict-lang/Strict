using Strict.Expressions;
using System.Globalization;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ToEvaluator(Executor executor)
{
	public ValueInstance Evaluate(To to, ExecutionContext ctx)
	{
		executor.Statistics.ToConversionCount++;
		var left = executor.RunExpression(to.Instance!, ctx);
		if (to.Instance!.ReturnType.IsText && to.ConversionType.IsNumber && left.IsText)
			return new ValueInstance(to.ConversionType,
				double.Parse(left.Text, CultureInfo.InvariantCulture));
		if (to.ConversionType.IsText)
			return executor.CreateValueInstance(to.ConversionType, left.ToExpressionCodeString());
		if (to.Method.IsTrait)
			throw new ToMethodNotImplemented(left, to.ConversionType);
		return executor.EvaluateMethodCall(to, ctx);
	}

	public sealed class ToMethodNotImplemented(ValueInstance left, Type toConversionType)
		: ExecutionFailed(toConversionType,
			"Conversion from " + left + " to " + toConversionType.Name + " not supported");
}