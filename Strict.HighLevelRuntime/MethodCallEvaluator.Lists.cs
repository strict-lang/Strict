using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed partial class MethodCallEvaluator
{
	private static ValueInstance? ConvertToListValue(ValueInstance value)
	{
		if (value.IsList)
			return value;
		var typeInstance = value.TryGetValueTypeInstance();
		if (typeInstance == null)
			return null;
		if (typeInstance.TryGetValue(Type.ElementsLowercase, out var elements))
			return elements.IsList
				? elements
				// ReSharper disable TailRecursiveCall
				: ConvertToListValue(elements);
		if (typeInstance.TryGetValue(Type.IteratorLowercase, out var iterator))
			return iterator.IsList
				? iterator
				: ConvertToListValue(iterator);
		if (!typeInstance.ReturnType.IsList)
			return null;
		var length = value.GetIteratorLength();
		var iteratorValues = new ValueInstance[length];
		var listItemType = typeInstance.ReturnType.GetFirstImplementation();
		var characterType = listItemType.GetType(Type.Character);
		for (var index = 0; index < length; index++)
			iteratorValues[index] = value.GetIteratorValue(characterType, index);
		return new ValueInstance(typeInstance.ReturnType, iteratorValues);
	}

	private static bool HasListType(ValueInstance value) => !value.IsText && value.GetType().IsList;

	private static bool IsEmptyListTypeCheck(ValueInstance left, ValueInstance right) =>
		(!left.IsList || left.List.Count == 0) && (!right.IsList || right.List.Count == 0);

	private ValueInstance CombineLists(ValueInstance leftList, List<ValueInstance> rightList,
		ExecutionContext ctx, MethodCall call)
	{
		var leftItemType = leftList.List.ReturnType.GetFirstImplementation();
		var convertedRightItems = new ValueInstance[rightList.Count];
		for (var rightItemIndex = 0; rightItemIndex < rightList.Count; rightItemIndex++)
		{
			convertedRightItems[rightItemIndex] = RightItemForCombineLists(leftItemType,
				rightList[rightItemIndex], ctx, call);
			if (convertedRightItems[rightItemIndex].IsError)
				return convertedRightItems[rightItemIndex];
		}
		if (leftList.IsMutable)
		{
			foreach (var item in convertedRightItems)
				leftList.List.Items.Add(item);
			return leftList;
		}
		return new ValueInstance(leftList.List.Appended(convertedRightItems));
	}

	private ValueInstance RightItemForCombineLists(Type leftItemType, ValueInstance item,
		ExecutionContext ctx, MethodCall call)
	{
		if (leftItemType.IsText && !item.IsText)
			return new ValueInstance(item.ToExpressionCodeString());
		if (leftItemType.IsNumber && (item.IsText || item.IsPrimitiveType(interpreter.characterType)))
			return double.TryParse(item.ToExpressionCodeString(), out var itemNumber)
				? new ValueInstance(leftItemType, itemNumber)
				: Error(
					"Cannot downcast Text to Number for list: " +
					item.ToString().Replace("\"", "\\\"", StringComparison.Ordinal), ctx, call);
		return item;
	}

	private static ValueInstance SubtractLists(ValueInstance leftList, List<ValueInstance> rightList)
	{
		if (leftList.IsMutable)
		{
			foreach (var rightItem in rightList)
			{
				var removeIndex = leftList.List.Items.FindIndex(leftItem => leftItem.Equals(rightItem));
				if (removeIndex >= 0)
					leftList.List.Items.RemoveAt(removeIndex);
			}
			return leftList;
		}
		var removed = new bool[rightList.Count];
		var temp = new ValueInstance[leftList.List.Items.Count];
		var itemCount = 0;
		for (var leftIndex = 0; leftIndex < leftList.List.Items.Count; leftIndex++)
		{
			var shouldKeep = true;
			for (var rightIndex = 0; rightIndex < rightList.Count; rightIndex++)
				if (!removed[rightIndex] && leftList.List.Items[leftIndex].Equals(rightList[rightIndex]))
				{
					removed[rightIndex] = true;
					shouldKeep = false;
					break;
				}
			if (shouldKeep)
				temp[itemCount++] = leftList.List.Items[leftIndex];
		}
		var result = new ValueInstance[itemCount];
		Array.Copy(temp, result, itemCount);
		return new ValueInstance(leftList.List.ReturnType, result);
	}

	private static ValueInstance AddToList(ValueInstance leftList, ValueInstance right)
	{
		var isLeftText = leftList.List.ReturnType is GenericTypeImplementation
		{
			Generic.Name: Type.List
		} list && list.ImplementationTypes[0].IsText;
		var rightItem = isLeftText && !right.IsText
			? new ValueInstance(right.ToExpressionCodeString())
			: right;
		if (leftList.IsMutable)
		{
			leftList.List.Items.Add(rightItem);
			return leftList;
		}
		return new ValueInstance(leftList.List.Appended([rightItem]));
	}

	private static ValueInstance RemoveFromList(ValueInstance leftList, ValueInstance right)
	{
		if (leftList.IsMutable)
		{
			leftList.List.Items.RemoveAll(item => item.Equals(right));
			return leftList;
		}
		var count = 0;
		for (var index = 0; index < leftList.List.Items.Count; index++)
			if (!leftList.List.Items[index].Equals(right))
				count++;
		var result = new ValueInstance[count];
		var resultIndex = 0;
		for (var index = 0; index < leftList.List.Items.Count; index++)
			if (!leftList.List.Items[index].Equals(right))
				result[resultIndex++] = leftList.List.Items[index];
		return new ValueInstance(leftList.List.ReturnType, result);
	}

	private static ValueInstance MultiplyLists(Type leftListType, Type numberType,
		List<ValueInstance> leftList, List<ValueInstance> rightList)
	{
		var result = new ValueInstance[leftList.Count];
		for (var index = 0; index < leftList.Count; index++)
			result[index] = new ValueInstance(numberType,
				leftList[index].GetArithmeticNumber() * rightList[index].GetArithmeticNumber());
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideLists(Type leftListType, Type numberType,
		List<ValueInstance> leftList, List<ValueInstance> rightList)
	{
		var result = new ValueInstance[leftList.Count];
		for (var index = 0; index < leftList.Count; index++)
			result[index] = new ValueInstance(numberType,
				leftList[index].GetArithmeticNumber() / rightList[index].GetArithmeticNumber());
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance MultiplyList(Type leftListType, List<ValueInstance> leftList,
		double rightNumber)
	{
		var result = new ValueInstance[leftList.Count];
		for (var i = 0; i < leftList.Count; i++)
			result[i] = new ValueInstance(leftList[i].GetType(),
				leftList[i].GetArithmeticNumber() * rightNumber);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideList(Type leftListType, List<ValueInstance> leftList,
		double rightNumber)
	{
		var result = new ValueInstance[leftList.Count];
		for (var i = 0; i < leftList.Count; i++)
			result[i] = new ValueInstance(leftList[i].GetType(),
				leftList[i].GetArithmeticNumber() / rightNumber);
		return new ValueInstance(leftListType, result);
	}
}
