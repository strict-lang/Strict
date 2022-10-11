using NUnit.Framework;
using static Strict.VM.VirtualMachine;

namespace Strict.VM.Tests
{
	public class ArithmeticTests
	{
		private VirtualMachine vm = null!;

		[SetUp]
		public void Setup() => vm = new VirtualMachine();

		[Test]
		public void SimpleArithmeticAddition()
		{
			var instructions = new Instruction[]
			{
				new(OperationCode.Push, 5), new(OperationCode.Push, 10), new(OperationCode.Add),
				new(OperationCode.Push, 3), new(OperationCode.Divide), new(OperationCode.Push, 1),
				new(OperationCode.Subtract), new(OperationCode.Push, 5), new(OperationCode.Multiply),
				new(OperationCode.Quit)
			};
			vm.Run(instructions);
			Assert.That(vm.register.Stack.Pop(), Is.EqualTo(20));
		}

		[Test]
		public void StackElementDoesNotExistWhenInstructionIsCalled()
		{
			var instructions = new Instruction[] { new(OperationCode.Add), new(OperationCode.Quit) };
			Assert.That(() => vm.Run(instructions),
				Throws.InstanceOf<StackElementDoesNotExistWhenInstructionIsCalled>());
		}
	}
}