namespace Strict.VirtualMachine;

public enum Instruction
{
	Set,
	StoreVariable,
	StoreFromRegister,
	Load,
	LoadConstant,
	LoopBegin,
	IterationEnd,
	SetLoadSeparator = 100,
	Add,
	Subtract,
	Multiply,
	Divide,
	Modulo,
	BinaryOperatorsSeparator = 200,
	GreaterThan,
	LessThan,
	Equal,
	NotEqual,
	ConditionalSeparator = 300,
	JumpIfTrue,
	JumpIfFalse,
	JumpIfNotZero,
	JumpEnd,
	JumpToIdIfFalse,
	JumpToIdIfTrue,
	JumpsSeparator = 400,
	Invoke,
	WriteToList,
	Return
}

