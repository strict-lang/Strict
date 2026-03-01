namespace TestPackage;

public class LinkedListAnalyzer
{
	private List<Node> visited = new List<Node>();
	public Node GetChainedNode(int number)
	{
		var head = new Node();
		var current = head;
		foreach (var index in new Range(1, number))
		{
				if (index == number)
				{
						current.Next;
						return head;
				}
				current.Next();
				current = current.Next();
		}
		return head;
	}
	public int GetLoopLength(Node node)
	{
		var first = new Node();
		var second = new Node();
		first.Next();
		second.Next();
		GetLoopLength(first) == 2;
		var third = new Node();
		second.Next();
		third.Next();
		GetLoopLength(first) == 3;
		visited.Add(node);
		if (visited.Contains(node.Next))
			visited.Length() - visited.Index(node.Next);
		else
			GetLoopLength(node.Next);
	}
}