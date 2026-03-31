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
		if (ShouldUseGenericConversion(left, to) &&
			TryConvert(left, to.ConversionType, out var convertedValue))
			return convertedValue;
		return to.Method.IsTrait
			? throw new ToMethodNotImplemented(left, to.ConversionType)
			: interpreter.methodCallEvaluator.Evaluate(to, ctx);
	}

	private static bool ShouldUseGenericConversion(ValueInstance value, To to) =>
		!to.ConversionType.IsText || to.Method.IsTrait || value.TryGetValueTypeInstance() == null ||
		to.Method.Type != value.GetType();

	private static bool TryConvert(ValueInstance value, Type conversionType,
		out ValueInstance convertedValue)
	{
		if (value.IsText)
		{
			if (conversionType.IsText)
			{
				convertedValue = value;
				return true;
			}
			if (conversionType.IsNumber && double.TryParse(value.Text, CultureInfo.InvariantCulture,
				out var parsedNumber))
			{
				convertedValue = new ValueInstance(conversionType, parsedNumber);
				return true;
			}
		}
		if (value.IsList && conversionType.IsList)
		{
			var convertedItems = new ValueInstance[value.List.Items.Count];
			var targetItemType = conversionType.GetFirstImplementation();
			for (var index = 0; index < convertedItems.Length; index++)
				if (!TryConvert(value.List.Items[index], targetItemType, out convertedItems[index]))
				{
					convertedValue = default;
					return false;
				}
			convertedValue = new ValueInstance(conversionType, convertedItems);
			return true;
		}
		if (conversionType.IsText)
		{
			convertedValue = CreateTextValue(value);
			return true;
		}
		if (conversionType.IsNumber &&
			(value.IsText || value.IsPrimitiveType(conversionType.GetType(Type.Character))))
		{
			if (double.TryParse(value.ToExpressionCodeString(), CultureInfo.InvariantCulture,
				out var numberValue))
			{
				convertedValue = new ValueInstance(conversionType, numberValue);
				return true;
			}
			convertedValue = default;
			return false;
		}
		if (value.IsSameOrCanBeUsedAs(conversionType))
		{
			convertedValue = new ValueInstance(value, conversionType);
			return true;
		}
		convertedValue = default;
		return false;
	}

	private static ValueInstance CreateTextValue(ValueInstance value) =>
		value.TryGetValueTypeInstance() is { } typeInstance
			? new ValueInstance(typeInstance.ToAutomaticText())
			: new ValueInstance(value.ToExpressionCodeString());

	public sealed class ToMethodNotImplemented(ValueInstance left, Type toConversionType)
		: InterpreterExecutionFailed(toConversionType, "Conversion from " + left + " to " +
			toConversionType.Name + " not supported");
}