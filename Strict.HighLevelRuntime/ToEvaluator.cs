using Strict.Expressions;
using Strict.Language;
using System.Globalization;

namespace Strict.HighLevelRuntime;

internal sealed class ToEvaluator(Executor executor)
{
	public ValueInstance Evaluate(To to, ExecutionContext ctx)
	{
		executor.Statistics.ToConversionCount++;
		var left = executor.RunExpression(to.Instance!, ctx).Value;
		if (to.Instance!.ReturnType.Name == Base.Text && to.ConversionType.Name == Base.Number &&
			left is string textValue)
			return new ValueInstance(to.ConversionType,
				double.Parse(textValue, CultureInfo.InvariantCulture), executor.Statistics);
		if (to.ConversionType.Name == Base.Text)
			return new ValueInstance(to.ConversionType, left?.ToString() ?? "", executor.Statistics);
		if (!to.Method.IsTrait && to.Method.Type.Name != Base.Number)
			return executor.EvaluateMethodCall(to, ctx);
		return !to.Method.IsTrait
			? executor.EvaluateMethodCall(to, ctx)
			: throw new NotSupportedException("Conversion to " + to.ConversionType.Name + " not supported");
	}
}
