namespace Strict.Runtime;

/// <summary>
/// Each Instruction corresponds to a Statement class in the <see cref="Statements" /> namespace
/// here. For details on what each instruction does, see the corresponding Statement class.
/// </summary>
public enum Instruction
{
	LoadConstantToRegister,
	LoadVariableToRegister,
	StoreConstantToVariable,
	StoreRegisterToVariable,
	Set,
	LoopBegin,
	LoopBeginRange,
	IterationEnd,
	ListCall,
	WriteToList,
	WriteToTable,
	RemoveFromTable,
	Remove,
	ToText,
	ToNumber,
	StoreSeparator = 100,
	Add,
	Subtract,
	Multiply,
	Divide,
	Modulo,
	ArithmeticSeparator = 200,
	Equal,
	NotEqual,
	LessThan,
	GreaterThan,
	BinaryOperatorsSeparator = 300,
	Jump,
	JumpIfTrue,
	JumpIfFalse,
	JumpIfNotZero,
	JumpEnd,
	JumpToIdIfFalse,
	JumpToIdIfTrue,
	ControlFlowSeparator = 400,
	Invoke,
	Return
}

