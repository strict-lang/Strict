using Strict.Expressions;
using Strict.Language;
using System.Globalization;

namespace Strict.HighLevelRuntime;

internal sealed class ToEvaluator(Executor executor)
{
	public ValueInstance Evaluate(To to, ExecutionContext ctx)
	{
		executor.Statistics.ToConversionCount++;
		var left = executor.RunExpression(to.Instance!, ctx);
		if (to.Instance!.ReturnType.IsText && to.ConversionType.IsNumber &&
			left.IsText)
     return new ValueInstance( executor.Number(to.ConversionType,
				double.Parse(textValue, CultureInfo.InvariantCulture));
		if (to.ConversionType.IsText)
			return executor.CreateValueInstance(to.ConversionType, left?.ToString() ?? "");
		if (to.Method is { IsTrait: false, Type.IsNumber: false })
			return executor.EvaluateMethodCall(to, ctx);
		return !to.Method.IsTrait
			? executor.EvaluateMethodCall(to, ctx)
			: throw new NotSupportedException("Conversion to " + to.ConversionType.Name + " not supported");
	}
}
