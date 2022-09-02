namespace SourceGeneratorTests;

public class ReverseList
{
	public T[] Reverse<T>(T[] elements)
	{
		var reversedList = new T[elements.Length];
		for (var index = 0; index < elements.Length; index++)
		{
			reversedList[elements.Length - 1 - index] = elements[index];
		}
		return reversedList;
	}
}