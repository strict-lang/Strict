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
		if (TryConvert(left, to.ConversionType, out var convertedValue))
			return convertedValue;
		return to.Method.IsTrait
			? throw new ToMethodNotImplemented(left, to.ConversionType)
			: interpreter.methodCallEvaluator.Evaluate(to, ctx);
	}

	internal static bool TryConvert(ValueInstance value, Type conversionType,
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
			? new ValueInstance(BuildMembersText(typeInstance))
			: new ValueInstance(value.ToExpressionCodeString());

	private static string BuildMembersText(ValueTypeInstance typeInstance)
	{
		var values = typeInstance.Values;
		if (values.Length == 0)
			return typeInstance.ReturnType.Name; //ncrunch: no coverage
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