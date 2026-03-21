namespace TestPackage;

public class LinkedListAnalyzer
{
	private List<Node> visited = new List<Node>();
	public Node GetChainedNode(int number)
	{
		var head = new Node();
		var current = head;
		foreach (var index in new Range(1, number))
				Next = Node;
				current = current.Next;
		Next = head;
	}
	public int GetLoopLength(Node node)
	{
		var first = new Node();
		var second = new Node();
		Next = second;
		Next = first;
		GetLoopLength(first) == 2;
		var third = new Node();
		Next = third;
		Next = first;
		GetLoopLength(first) == 3;
		visited.Add(node);
		if (visited in node.Next)
			visited.Length();
		else
			GetLoopLength(node.Next);
	}
}