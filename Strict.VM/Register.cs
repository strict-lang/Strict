namespace Strict.VM;

public sealed class Register
{
	public Stack<int> Stack { get; } = new();
	public int CurrentInstructionIndex { get; set; }
}