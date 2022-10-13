namespace Strict.VirtualMachine;

public enum Instruction
{
	Set,
	Add,
	Subtract,
	Multiply,
	Divide,
	BinaryOperatorsSeparator = 100,
	GreaterThan,
	LessThan,
	Equal,
	NotEqual,
	ConditionalSeparator = 200,
	JumpIfTrue,
	JumpIfFalse,
	JumpIfNotZero,
	JumpsSeparator = 300
}