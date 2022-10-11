namespace Strict.VM
{
	public class VirtualMachine
	{
		public readonly Register register = new();

		// ReSharper disable once MethodTooLong
		public void Run(Instruction[] instructions)
		{
			while (register.CurrentInstructionIndex != -1)
				switch (instructions[register.CurrentInstructionIndex].OperationCode)
				{
				case OperationCode.Add:
					AddInstruction();
					break;
				case OperationCode.Push:
					PushInstruction(instructions[register.CurrentInstructionIndex].Value);
					break;
				case OperationCode.Quit:
					QuitInstruction();
					break;
				case OperationCode.Subtract:
					SubtractInstruction();
					break;
				case OperationCode.Multiply:
					MultiplyInstruction();
					break;
				case OperationCode.Divide:
					DivideInstruction();
					break;
				}
		}

		private void AddInstruction()
		{
			var (right, left) = GetOperands();
			register.Stack.Push(right + left);
			++register.CurrentInstructionIndex;
		}

		private void SubtractInstruction()
		{
			var (right, left) = GetOperands();
			register.Stack.Push(left - right);
			++register.CurrentInstructionIndex;
		}

		private void MultiplyInstruction()
		{
			var (right, left) = GetOperands();
			register.Stack.Push(right * left);
			++register.CurrentInstructionIndex;
		}

		private void DivideInstruction()
		{
			var (right, left) = GetOperands();
			register.Stack.Push(left / right);
			++register.CurrentInstructionIndex;
		}

		private void PushInstruction(int value)
		{
			register.Stack.Push(value);
			++register.CurrentInstructionIndex;
		}

		private (int, int) GetOperands()
		{
			if (register.Stack.TryPop(out var right) && register.Stack.TryPop(out var left))
				return (right, left);
			throw new StackElementDoesNotExistWhenInstructionIsCalled();
		}

		private void QuitInstruction() => register.CurrentInstructionIndex = -1;
		public sealed class StackElementDoesNotExistWhenInstructionIsCalled : Exception { }
	}
}