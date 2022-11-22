namespace Strict.VirtualMachine;

public enum Instruction
{
	Set,
	StoreVariable,
	StoreConstant,
	Load,
	LoadConstant,
	InitLoopStatement,
	SetLoadSeparator = 100,
	Add,
	Subtract,
	Multiply,
	Divide,
	BinaryOperatorsSeparator = 200,
	GreaterThan,
	LessThan,
	Equal,
	NotEqual,
	ConditionalSeparator = 300,
	JumpIfTrue,
	JumpIfFalse,
	JumpIfNotZero,
	JumpsSeparator = 400,
	Return
}